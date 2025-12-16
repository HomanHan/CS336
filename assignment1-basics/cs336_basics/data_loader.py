import os
import torch
import numpy as np
import numpy.typing as npt


def get_dataset_memmap(path, dtype=np.uint16) -> npt.NDArray:
    """Load dataset using memory mapping for efficiency"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    dataset = np.memmap(path, dtype=dtype, mode="r")
    return dataset


def data_loading(
    x: npt.NDArray, b: int, m: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch for language-modelling from a 1D dataset.

    Args:
        x: A 1D numpy array of token ids.
        b: Batch size.
        m: Context length (sequence length) for each sample.
        device: Torch device string (e.g. 'cpu' or 'cuda:0').

    Returns:
        (x_batch, y_batch): Tuple of `torch.LongTensor` with shape `(b, m)` placed on `device`.
    """
    # If a path-like object was provided, memory-map the array from disk.

    n = x.shape[0]
    if n < m + 1:
        raise ValueError("dataset is too small for the requested context length")

    # Valid starting indices are [0, n - m - 1] inclusive such that y has length m
    max_start = n - m
    starts = np.random.randint(
        0, max_start, size=b
    )  # 随机采样，抽取 batch size 个起始位置 shape (b,)

    # Build batches by slicing. For memmapped arrays this avoids loading the whole file.
    x_batch = np.stack([x[s : s + m] for s in starts], axis=0)  # shape (b, m)
    y_batch = np.stack([x[s + 1 : s + 1 + m] for s in starts], axis=0)

    x_tensor = torch.from_numpy(x_batch).long().to(device)
    y_tensor = torch.from_numpy(y_batch).long().to(device)

    return (x_tensor, y_tensor)
