import torch
import torch.nn as nn
import einops


# 注意在第 i 个维度上进行计算（max, mean...），是对平行于该维度的每一个切片分别进行计算
# 即控制住其他维度不变。该维度大小为多少，每个切片中就有多少个元素参与计算。最终所有元素都会参与到计算
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
