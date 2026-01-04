import torch

def weighted_sum(x, weights):
    # x: n-dim [..., D]
    # weights: 1-dim [D]
    # returns: n-dim [...]
    return torch.sum(x * weights, dim=-1)

x = torch.randn(2, 3, 4)
weights = torch.randn(4)

print("x:", x)
print("weights:", weights)
print("shape:", weighted_sum(x, weights).shape)
print("weighted sum:", weighted_sum(x, weights))