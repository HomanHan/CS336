import torch
import torch.nn as nn
import os
import typing


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    """
    Save the model and optimizer state to a checkpoint file.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer whose state to save.
        iteration: The current training iteration (used for naming the checkpoint).
        out: The output directory where the checkpoint will be saved.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Load the model and optimizer state from a checkpoint file.

    Args:
        src: The source path of the checkpoint file.
        model: The PyTorch model to load the state into.
        optimizer: The optimizer to load the state into.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration
