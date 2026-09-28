"""Simulated weight-only quantization.

Every nn.Linear weight is rounded to a symmetric integer grid and stored back dequantized,
so the model computes exactly what a W8A16 / W4A16 kernel would, without needing one.
INT8 uses one scale per output channel; INT4 and below one scale per group of 128 inputs, as
GPTQ- and AWQ-style deployments do.
"""

import torch
from torch import nn


@torch.no_grad()
def _quantize(w: torch.Tensor, bits: int, group: int | None) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    out_f, in_f = w.shape
    x = w.float()
    if group and in_f % group == 0:
        x = x.reshape(out_f, in_f // group, group)
    else:
        x = x.reshape(out_f, 1, in_f)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = (x / scale).round().clamp(-qmax - 1, qmax)
    return (q * scale).reshape(out_f, in_f).to(w.dtype)


@torch.no_grad()
def fake_quantize_(model: nn.Module, bits: int) -> dict:
    group = 128 if bits <= 4 else None
    n, params, err = 0, 0, 0.0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight
            q = _quantize(w, bits, group)
            err += (q.float() - w.float()).pow(2).sum().item()
            params += w.numel()
            w.copy_(q)
            n += 1
    return {"bits": bits, "group": group, "linear_layers": n, "params": params, "mse": err / max(params, 1)}
