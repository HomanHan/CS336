import torch


# see Obsidian CS336 Assignments: Cap.1 Note: Cross Entropy Loss
def cross_entropy_loss(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross-entropy loss between logits and labels.
    Args:
        logits (torch.Tensor): The unnormalized predicted logits of shape (batch_size, vocab_size).
        label (torch.Tensor): The true labels of shape (batch_size,).
    Returns:
        Float[Tensor, ""]: The computed cross-entropy loss.
    """

    # Subtract the maximum logits for numerical stability
    max_logits = torch.max(logits, dim=-1, keepdim=True).values  # shape (batch_size, 1)
    stabilized_logits = logits - max_logits  # shape (batch_size, vocab_size)

    # Compute the log-sum-exp for each example
    log_sum_exp = torch.log(
        torch.sum(torch.exp(stabilized_logits), dim=-1, keepdim=True)
    )  # shape (batch_size, 1)

    # get the logits corresponding to the true labels
    chosen_logits = torch.gather(
        stabilized_logits, dim=-1, index=label.unsqueeze(-1)
    )  # shape (batch_size, 1)

    # # or use `torch.logsumexp`
    # log_sum_exp = torch.logsumexp(logits, dim=-1, keepdim=True)  # shape (batch_size, 1)
    # chosen_logits = torch.gather(
    #     logits, dim=-1, index=label.unsqueeze(-1)
    # )  # shape (batch_size, 1)

    # Compute the cross-entropy loss
    loss = -(chosen_logits - log_sum_exp)  # shape (batch_size, 1)

    return loss.mean()  # Return the mean loss over the batch. shape (batch_size,)
