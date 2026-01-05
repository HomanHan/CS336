import torch
import triton
import triton.language as tl
import einops


def weighted_sum(x, weights):
    # x: n-dim [..., D]
    # weights: 1-dim [D]
    # returns: n-dim [...]
    return torch.sum(x * weights, dim=-1)


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,
    output_ptr,
    x_stride_row,
    x_stride_dim,
    weight_stride_dim,  # likely 1
    output_stride_row,  # likely 1
    ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    # Each instance will compute the weighted sum of a tile of rows of x.
    # `tl.program_id` gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)

    # Block pointers give us a way to select from an ND region of memory
    # and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the memory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to use load/store at a time
    # - The order of the dimensions in memory from major to minor，常常根据 strides 的大小排序得到
    # axes (= np.argsort(strides)) for optimizations, especially useful on H100
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(0, 1),
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # Initialize a buffer to write to
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load the current block pointer
        # Since ROWS_TILE_SIZE might not divide ROWS, and D_TILE_SIZE might not divide D,
        # we need boundary checks for both dimensions
        row = tl.load(
            x_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(
            weight_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (D_TILE_SIZE,)

        # Compute the weighted sum of the row.
        output += tl.sum(
            row * weight[None, :], axis=1
        )  # weight[None, :] 把 weight 扩展成 (1, D_TILE_SIZE)，后面再用广播机制与 row 相乘， axis=1 对下标为 1 的维度求和

        # Move the pointers to the next tile.
        # These are (rows, columns) coordinate deltas
        x_block_ptr = x_block_ptr.advance(
            (0, D_TILE_SIZE)
        )  # Move by D_TILE_SIZE in the last dimension
        weight_block_ptr = weight_block_ptr.advance(
            (D_TILE_SIZE,)
        )  # Move by D_TILE_SIZE

    # Write output to the output block pointer (a single scalar per row).
    # Since ROWS_TILE_SIZE might not divide ROWS, we need boundary checks
    # 越界的部分会跳过，不会写入
    tl.store(
        output_block_ptr,
        output,
        boundary_check=(0,),
    )


# TODO: implement backward pass
@triton.jit
def weighted_sum_backward(
    x_ptr,
    weight_ptr,  # Input
    grad_output_ptr,  # Grad input
    grad_x_ptr,
    partial_grad_weight_ptr,  # Grad outputs
    stride_xr,
    stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr,
    stride_gxd,
    stride_gwb,
    stride_gwd,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    raise NotImplementedError("Backward pass not implemented yet")


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight) -> torch.Tensor:
        # Cache x and weight to be used in the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        D, output_dims = x.shape[-1], x.shape[:-1]

        # Reshape input tensor to 2D [*, D]，铺平到二维
        input_shape = x.shape
        x = einops.rearrange(x, "... d -> (...) d")  # e.g. (2, 3, 4) -> (6, 4)
        ctx.save_for_backward(x, weight)
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.D_TILE_SIZE = (
            triton.next_power_of_2(D) // 16
        )  # Roughly 16 loops through the embedding dimension
        ctx.ROWS_TILE_SIZE = 16  # Each thread processes 16 batch elements at a time
        ctx.input_shape = input_shape

        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        y = torch.empty(output_dims, device=x.device)
        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()

        # kernel_name[grid](*args, **kwargs)
        # [grid] 是一个 tuple，指定启动的 Triton Program 数量
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x,
            weight,
            y,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        print("shape of y in weighted_sum_fwd:", y.shape)
        return y.view(
            input_shape[:-1]
        )  # view 会调整 shape，保证输出 y 与输入 x[:-1] 的形状一致

    # TODO: implement backward pass
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("Backward pass not implemented yet")


x = torch.randn(32, 96, 128)
# 为什么数量级较小的时候会 floating point exception? e.g. (2, 4)
# A1: line 139 `ctx.D_TILE_SIZE = (triton.next_power_of_2(D) // 16)` 当 D 较小时会变成 0，导致后续 cdiv(D, D_TILE_SIZE) 出现除以 0 的错误

weights = torch.randn(128)

# torch implementation
print("x:", x)
print("weights:", weights)
print("shape:", weighted_sum(x, weights).shape)
print("weighted sum:", weighted_sum(x, weights))

# triton implementation
result = WeightedSumFunc.apply(x.cuda(), weights.cuda())
print("result shape:", result.shape)
print("result:", result)
