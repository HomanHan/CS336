import torch
import triton
import triton.language as tl


def triton_softmax(x: torch.Tensor):
    # Allocate output tensor
    y = torch.empty_like(x)
    # Determine grid
    M, N = x.shape  # Number of rows x number of columns
    block_size = triton.next_power_of_2(
        N
    )  # Each block contains all the columns, 对每一行做 softmax
    num_blocks = M  # Each block is a row
    # Launch kernel
    triton_softmax_kernel[(M,)](
        x_ptr=x,
        y_ptr=y,
        x_row_stride=x.stride(0),
        y_row_stride=y.stride(0),
        num_cols=N,
        BLOCK_SIZE=block_size,
    )

    return y


@triton.jit
def triton_softmax_kernel(
    x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr
):
    assert num_cols <= BLOCK_SIZE
    # Process each row independently
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Read from global memory
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(
        x_ptrs, mask=col_offsets < num_cols, other=float("-inf")
    )  # mask 将越界部分（> N）设为 -inf
    # Compute
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator
    # Write back to global memory
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)


x = torch.randn(4, 5, device="cuda")
y_triton = triton_softmax(x)
y_torch = torch.nn.functional.softmax(x, dim=-1)
print("Triton Softmax:\n", y_triton)
print("PyTorch Softmax:\n", y_torch)
print("Difference:\n", y_triton - y_torch)
