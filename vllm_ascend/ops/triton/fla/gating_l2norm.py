# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.fla.utils import _gdn_gating_rows
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def _l2norm_strided_rows(
    X,
    Y,
    eps,
    num_rows,
    num_heads,
    token_stride,
    head_stride,
    row_iter,
    core_id,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    dims = tl.arange(0, BLOCK_DIM)[None, :]
    for chunk in range(row_iter):
        rows = (core_id * row_iter + chunk) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        offsets = (rows // num_heads) * token_stride + (rows % num_heads) * head_stride + dims
        mask = (rows < num_rows) & (dims < HEAD_DIM)
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] + eps)
        tl.store(Y + rows * HEAD_DIM + dims, y, mask=mask)


@triton.jit(
    do_not_specialize=[
        "num_tokens", "num_qk_rows", "num_qk_heads", "num_v_heads",
        "q_token_stride", "q_head_stride", "k_token_stride", "k_head_stride",
        "gating_programs", "norm_programs", "gating_iters", "norm_iters", "eps",
    ]
)
def _gating_l2norm_qk_kernel(
    Q, K, YQ, YK, G, BETA, A_LOG, A, B, DT_BIAS,
    num_tokens, num_qk_rows, num_qk_heads, num_v_heads,
    q_token_stride, q_head_stride, k_token_stride, k_head_stride,
    gating_programs, norm_programs, gating_iters, norm_iters, eps,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    # Gating covers every token; the two independent norm segments only cover
    # the decode Q/K views. No segment consumes another segment's output.
    pid = tl.program_id(0)
    if pid < gating_programs:
        _gdn_gating_rows(
            G, BETA, A_LOG, A, B, DT_BIAS, num_v_heads, num_tokens,
            1.0, 20.0, BLOCK_HEADS, BLOCK_TOKENS, gating_iters, pid,
        )
    elif pid < gating_programs + norm_programs:
        _l2norm_strided_rows(
            Q, YQ, eps, num_qk_rows, num_qk_heads, q_token_stride, q_head_stride,
            norm_iters, pid - gating_programs, HEAD_DIM, BLOCK_DIM, BLOCK_ROWS,
        )
    else:
        _l2norm_strided_rows(
            K, YK, eps, num_qk_rows, num_qk_heads, k_token_stride, k_head_stride,
            norm_iters, pid - gating_programs - norm_programs, HEAD_DIM, BLOCK_DIM, BLOCK_ROWS,
        )


def gating_l2norm_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute full-batch GDN gates and decode-only Q/K L2 norm in one launch.

    Q/K are views of shape ``[1, decode_tokens, key_heads, head_dim]`` with
    contiguous features. Token/head strides may differ, including transposed
    views of a mixed batch's ``[head, token, dim]`` convolution output. Outputs
    Q/K are contiguous in token-first order; the input and prefill rows are not
    modified. A/B cover the full batch with shape ``[tokens, value_heads]``.
    """
    if q.ndim != 4 or q.shape[0] != 1 or q.shape != k.shape or q.dtype != k.dtype:
        raise ValueError("gating_l2norm_qk: q/k must have matching [1, tokens, heads, dim] shape and dtype")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("gating_l2norm_qk: q/k features must be contiguous")
    if a.ndim != 2 or a.shape != b.shape or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("gating_l2norm_qk: a/b must be contiguous with matching [tokens, heads] shape")
    num_tokens, num_v_heads = a.shape
    _, num_decode_tokens, num_qk_heads, head_dim = q.shape
    if min(num_decode_tokens, num_qk_heads, head_dim, num_v_heads) <= 0 or num_decode_tokens > num_tokens:
        raise ValueError("gating_l2norm_qk: expected nonempty decode Q/K within the full gate batch")
    if A_log.numel() != num_v_heads or dt_bias.numel() != num_v_heads:
        raise ValueError("gating_l2norm_qk: A_log/dt_bias must contain one value per gate head")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("gating_l2norm_qk: A_log/dt_bias must be contiguous")
    if any(t.device != q.device for t in (k, A_log, a, b, dt_bias)):
        raise ValueError("gating_l2norm_qk: all inputs must be on the same device")
    max_feature_bytes = 65536
    if head_dim * q.element_size() > max_feature_bytes:
        raise ValueError("gating_l2norm_qk: features larger than 64 KiB are unsupported")

    g = torch.empty((1, num_tokens, num_v_heads), dtype=torch.float32, device=a.device)
    beta = torch.empty((1, num_tokens, num_v_heads), dtype=b.dtype, device=b.device)
    y_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    y_k = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    num_cores = get_vectorcore_num()
    block_rows, block_heads, block_tokens = 32, 8, 64
    num_qk_rows = num_decode_tokens * num_qk_heads
    gating_programs = min(num_cores, triton.cdiv(num_tokens, block_tokens))
    norm_programs = min(num_cores, triton.cdiv(num_qk_rows, block_rows))
    gating_iters = triton.cdiv(num_tokens, gating_programs * block_tokens)
    norm_iters = triton.cdiv(num_qk_rows, norm_programs * block_rows)
    _gating_l2norm_qk_kernel[(gating_programs + 2 * norm_programs,)](
        q, k, y_q, y_k, g, beta, A_log, a, b, dt_bias,
        num_tokens, num_qk_rows, num_qk_heads, num_v_heads,
        q.stride(1), q.stride(2), k.stride(1), k.stride(2),
        gating_programs, norm_programs, gating_iters, norm_iters, eps,
        HEAD_DIM=head_dim,
        BLOCK_DIM=triton.next_power_of_2(head_dim),
        BLOCK_ROWS=block_rows,
        BLOCK_HEADS=block_heads,
        BLOCK_TOKENS=block_tokens,
    )
    return g, beta, y_q, y_k

