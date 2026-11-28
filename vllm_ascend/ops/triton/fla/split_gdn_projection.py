# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Pack noninterleaved GDN QKVZ and BA projections in one launch."""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_BLOCK = 1024


@triton.jit
def _split_gdn_projection_kernel(
    qkvz_source,
    ba_source,
    qkv_out,
    z_out,
    b_out,
    a_out,
    QKVZ_ROW_STRIDE: tl.constexpr,
    BA_ROW_STRIDE: tl.constexpr,
    QKV_DIM: tl.constexpr,
    Z_DIM: tl.constexpr,
    BA_HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    columns = tile * BLOCK + tl.arange(0, BLOCK)

    qkv = tl.load(qkvz_source + row * QKVZ_ROW_STRIDE + columns, mask=columns < QKV_DIM, other=0)
    z = tl.load(qkvz_source + row * QKVZ_ROW_STRIDE + QKV_DIM + columns, mask=columns < Z_DIM, other=0)
    tl.store(qkv_out + row * QKV_DIM + columns, qkv, mask=columns < QKV_DIM)
    tl.store(z_out + row * Z_DIM + columns, z, mask=columns < Z_DIM)

    b = tl.load(ba_source + row * BA_ROW_STRIDE + columns, mask=columns < BA_HALF, other=0)
    a = tl.load(ba_source + row * BA_ROW_STRIDE + BA_HALF + columns, mask=columns < BA_HALF, other=0)
    tl.store(b_out + row * BA_HALF + columns, b, mask=columns < BA_HALF)
    tl.store(a_out + row * BA_HALF + columns, a, mask=columns < BA_HALF)


def _split_gdn_projection_impl(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    mixed_qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    qkv_dim: int,
    z_dim: int,
    ba_half: int,
) -> None:
    num_tokens = qkvz.shape[0]
    _split_gdn_projection_kernel[(num_tokens, triton.cdiv(max(qkv_dim, z_dim, ba_half), _BLOCK))](
        qkvz,
        ba,
        mixed_qkv,
        z,
        b,
        a,
        qkvz.stride(0),
        ba.stride(0),
        qkv_dim,
        z_dim,
        ba_half,
        _BLOCK,
    )


def _split_gdn_projection_fake(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    mixed_qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    qkv_dim: int,
    z_dim: int,
    ba_half: int,
) -> None:
    return None


direct_register_custom_op(
    op_name="ascend_split_gdn_projection",
    op_func=_split_gdn_projection_impl,
    mutates_args=["mixed_qkv", "z", "b", "a"],
    fake_impl=_split_gdn_projection_fake,
)


def split_gdn_projection(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    qkv_dim: int,
    z_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = qkvz.shape[0]
    ba_half = ba.shape[-1] // 2
    if num_tokens == 1 and qkvz.stride(-1) == 1 and ba.stride(-1) == 1:
        # Each slice of a single contiguous row is itself contiguous. Keep
        # the projections as views and avoid four allocations and a copy kernel.
        mixed_qkv = qkvz.narrow(-1, 0, qkv_dim)
        z = qkvz.narrow(-1, qkv_dim, z_dim).reshape(1, z_dim // head_v_dim, head_v_dim)
        b = ba.narrow(-1, 0, ba_half)
        a = ba.narrow(-1, ba_half, ba_half)
        return mixed_qkv, z, b, a

    mixed_qkv = torch.empty((num_tokens, qkv_dim), device=qkvz.device, dtype=qkvz.dtype)
    z = torch.empty((num_tokens, z_dim // head_v_dim, head_v_dim), device=qkvz.device, dtype=qkvz.dtype)
    b = torch.empty((num_tokens, ba_half), device=ba.device, dtype=ba.dtype)
    a = torch.empty_like(b)
    torch.ops.vllm.ascend_split_gdn_projection(qkvz, ba, mixed_qkv, z, b, a, qkv_dim, z_dim, ba_half)
    return mixed_qkv, z, b, a
