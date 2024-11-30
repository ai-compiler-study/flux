import os

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange


def _compiled_xformers_flash_hopper(q, k, v):
    import xformers.ops

    torch_custom_op_compile = os.getenv("TORCH_CUSTOM_OP_COMPILE", "0") == "1"

    if torch_custom_op_compile:
        xformers_flash3 = torch.compile(
            xformers.ops.fmha.flash3.FwOp,
            fullgraph=True,
            backend="inductor",
        )
    else:
        xformers_flash3 = xformers.ops.fmha.flash3.FwOp()
    softmax_scale = q.size(-1) ** -0.5

    return xformers.ops.fmha.memory_efficient_attention_forward(  # noqa: E731
        q,
        k,
        v,
        scale=softmax_scale,
        op=xformers_flash3,
    )

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    xformers_flash3 = os.getenv("XFORMERS_FLASH3", "0") == "1"
    torch_sdpa = os.getenv("TORCH_SDPA", "0") == "1"
    triton_attention = os.getenv("TRITON_ATTENTION", "0") == "1"

    q, k = apply_rope(q, k, pe)

    if xformers_flash3:
        if torch_sdpa or triton_attention:
            print(
                "Warning: xformers_flash3 is enabled, but torch_sdpa or triton_attention is also enabled. "
                "Please remain only one of them."
            )

        q = q.permute(0, 2, 1, 3) # B, H, S, D
        k = k.permute(0, 2, 1, 3) # B, H, S, D
        v = v.permute(0, 2, 1, 3) # B, H, S, D
        
        x = _compiled_xformers_flash_hopper(q, k, v).permute(0,2,1,3)
    elif torch_sdpa:
        if triton_attention:
            print(
                "Warning: torch_sdpa is enabled, but triton_attention is also enabled. "
                "Please remain only one of them."
            )

        x = scaled_dot_product_attention(q, k, v)
    elif triton_attention:
        from triton.ops import attention as attention_triton

        softmax_scale = q.size(-1) ** -0.5
        x = attention_triton(q, k, v, True, softmax_scale)
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
