import torch
import torch.nn as nn
import torch.nn.functional as F


def intermediate_correction_fn(expansion_ratio: float, hidden_size: int) -> int:
    return int(((expansion_ratio * hidden_size) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(hidden_size: int, expansion_ratio: float, dropout: float = 0.1, use_bias: bool = False):
    intermediate_size = intermediate_correction_fn(expansion_ratio, hidden_size)
    return nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, intermediate_size * 2, bias=use_bias),
        SwiGLU(),
        nn.Dropout(dropout),
        nn.Linear(intermediate_size, hidden_size, bias=use_bias),
    )
