# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

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
        "num_tokens",
        "num_qk_rows",
        "num_qk_heads",
        "num_v_heads",
        "q_token_stride",
        "q_head_stride",
        "k_token_stride",
        "k_head_stride",
        "gating_programs",
        "norm_programs",
        "gating_iters",
        "norm_iters",
        "eps",
        "state_inner_size",
        "state_row_stride",
        "state_col_blocks",
    ]
)
def _gating_l2norm_qk_kernel(
    Q,
    K,
    YQ,
    YK,
    G,
    BETA,
    A_LOG,
    A,
    B,
    DT_BIAS,
    num_tokens,
    num_qk_rows,
    num_qk_heads,
    num_v_heads,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    gating_programs,
    norm_programs,
    gating_iters,
    norm_iters,
    eps,
    STATE,
    STATE_INDICES,
    HAS_INITIAL_STATE,
    INITIAL_STATE,
    state_inner_size,
    state_row_stride,
    state_col_blocks,
    HAS_QK: tl.constexpr,
    HAS_STATE: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    # Gates cover the full batch, Q/K norm only covers decode, and gather/clear
    # only covers prefill states. All segments have disjoint outputs.
    pid = tl.program_id(0)
    if pid < gating_programs:
        _gdn_gating_rows(
            G,
            BETA,
            A_LOG,
            A,
            B,
            DT_BIAS,
            num_v_heads,
            num_tokens,
            1.0,
            20.0,
            BLOCK_HEADS,
            BLOCK_TOKENS,
            gating_iters,
            pid,
        )
    elif pid < gating_programs + 2 * norm_programs:
        if HAS_QK:
            if pid < gating_programs + norm_programs:
                _l2norm_strided_rows(
                    Q,
                    YQ,
                    eps,
                    num_qk_rows,
                    num_qk_heads,
                    q_token_stride,
                    q_head_stride,
                    norm_iters,
                    pid - gating_programs,
                    HEAD_DIM,
                    BLOCK_DIM,
                    BLOCK_ROWS,
                )
            else:
                _l2norm_strided_rows(
                    K,
                    YK,
                    eps,
                    num_qk_rows,
                    num_qk_heads,
                    k_token_stride,
                    k_head_stride,
                    norm_iters,
                    pid - gating_programs - norm_programs,
                    HEAD_DIM,
                    BLOCK_DIM,
                    BLOCK_ROWS,
                )
    else:
        if HAS_STATE:
            state_pid = pid - gating_programs - 2 * norm_programs
            row = state_pid // state_col_blocks
            cols = (state_pid % state_col_blocks) * BLOCK_STATE + tl.arange(0, BLOCK_STATE)
            mask = cols < state_inner_size
            dst = INITIAL_STATE + row.to(tl.int64) * state_inner_size + cols
            has_state = tl.load(HAS_INITIAL_STATE + row).to(tl.int1)
            if has_state:
                state_row = tl.load(STATE_INDICES + row).to(tl.int64)
                values = tl.load(STATE + state_row * state_row_stride + cols, mask=mask, other=0.0)
                tl.store(dst, values, mask=mask)
            else:
                # Fresh requests do not read a stale cache row only to zero it.
                tl.store(dst, tl.zeros((BLOCK_STATE,), dtype=INITIAL_STATE.dtype.element_ty), mask=mask)


def _gating_l2norm_qk(
    q: torch.Tensor | None,
    k: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    eps: float = 1e-6,
    ssm_state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
) -> tuple:
    if (q is None) != (k is None):
        raise ValueError("gating_l2norm_qk: q/k must both be present or both be None")
    if q is not None:
        if q.ndim != 4 or q.shape[0] != 1 or q.shape != k.shape or q.dtype != k.dtype:
            raise ValueError("gating_l2norm_qk: q/k must have matching [1, tokens, heads, dim] shape and dtype")
        if q.stride(-1) != 1 or k.stride(-1) != 1:
            raise ValueError("gating_l2norm_qk: q/k features must be contiguous")
    if a.ndim != 2 or a.shape != b.shape or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("gating_l2norm_qk: a/b must be contiguous with matching [tokens, heads] shape")
    num_tokens, num_v_heads = a.shape
    if num_tokens <= 0 or num_v_heads <= 0:
        raise ValueError("gating_l2norm_qk: gate inputs must be nonempty")
    num_decode_tokens, num_qk_heads, head_dim = 0, 1, 1
    if q is not None:
        _, num_decode_tokens, num_qk_heads, head_dim = q.shape
        if min(num_decode_tokens, num_qk_heads, head_dim) <= 0 or num_decode_tokens > num_tokens:
            raise ValueError("gating_l2norm_qk: expected nonempty decode Q/K within the full gate batch")
    if A_log.numel() != num_v_heads or dt_bias.numel() != num_v_heads:
        raise ValueError("gating_l2norm_qk: A_log/dt_bias must contain one value per gate head")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("gating_l2norm_qk: A_log/dt_bias must be contiguous")
    if any(t.device != a.device for t in (q, k, A_log, b, dt_bias) if t is not None):
        raise ValueError("gating_l2norm_qk: all inputs must be on the same device")
    max_feature_bytes = 65536
    if q is not None and head_dim * q.element_size() > max_feature_bytes:
        raise ValueError("gating_l2norm_qk: features larger than 64 KiB are unsupported")

    g = torch.empty((1, num_tokens, num_v_heads), dtype=torch.float32, device=a.device)
    beta = torch.empty((1, num_tokens, num_v_heads), dtype=b.dtype, device=b.device)
    y_q = torch.empty(q.shape, dtype=q.dtype, device=q.device) if q is not None else None
    y_k = torch.empty(k.shape, dtype=k.dtype, device=k.device) if k is not None else None
    initial_state = None
    state_inner_size, state_row_stride, state_col_blocks, state_programs = 0, 0, 0, 0
    block_state = 4096
    if ssm_state is not None:
        if ssm_state.ndim != 4 or not ssm_state.is_contiguous() or ssm_state.numel() == 0:
            raise ValueError("gating_l2norm_qk: state must be nonempty and contiguous [slots, heads, Dv, Dk]")
        if state_indices is None or has_initial_state is None:
            raise ValueError("gating_l2norm_qk: state indices and initial-state flags are required")
        if state_indices.ndim != 1 or state_indices.numel() == 0 or not state_indices.is_contiguous():
            raise ValueError("gating_l2norm_qk: state indices must be a nonempty contiguous vector")
        if state_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("gating_l2norm_qk: state indices must be int32 or int64")
        if (
            has_initial_state.shape != state_indices.shape
            or has_initial_state.dtype != torch.bool
            or not has_initial_state.is_contiguous()
        ):
            raise ValueError("gating_l2norm_qk: initial-state flags must be a matching contiguous bool vector")
        if any(t.device != a.device for t in (ssm_state, state_indices, has_initial_state)):
            raise ValueError("gating_l2norm_qk: state and metadata must be on the gate device")
        state_inner_size = math.prod(ssm_state.shape[1:])
        state_row_stride = ssm_state.stride(0)
        state_col_blocks = triton.cdiv(state_inner_size, block_state)
        state_programs = state_indices.numel() * state_col_blocks
        initial_state = torch.empty(
            (state_indices.numel(), *ssm_state.shape[1:]), dtype=ssm_state.dtype, device=ssm_state.device
        )
    num_cores = get_vectorcore_num()
    block_rows, block_heads, block_tokens = 32, 8, 64
    num_qk_rows = num_decode_tokens * num_qk_heads
    gating_programs = min(num_cores, triton.cdiv(num_tokens, block_tokens))
    norm_programs = min(num_cores, triton.cdiv(num_qk_rows, block_rows))
    gating_iters = triton.cdiv(num_tokens, gating_programs * block_tokens)
    norm_iters = triton.cdiv(num_qk_rows, norm_programs * block_rows) if norm_programs else 0
    _gating_l2norm_qk_kernel[(gating_programs + 2 * norm_programs + state_programs,)](
        q,
        k,
        y_q,
        y_k,
        g,
        beta,
        A_log,
        a,
        b,
        dt_bias,
        num_tokens,
        num_qk_rows,
        num_qk_heads,
        num_v_heads,
        q.stride(1) if q is not None else 0,
        q.stride(2) if q is not None else 0,
        k.stride(1) if k is not None else 0,
        k.stride(2) if k is not None else 0,
        gating_programs,
        norm_programs,
        gating_iters,
        norm_iters,
        eps,
        ssm_state,
        state_indices,
        has_initial_state,
        initial_state,
        state_inner_size,
        state_row_stride,
        state_col_blocks,
        HAS_QK=q is not None,
        HAS_STATE=ssm_state is not None,
        BLOCK_STATE=block_state,
        HEAD_DIM=head_dim,
        BLOCK_DIM=triton.next_power_of_2(head_dim),
        BLOCK_ROWS=block_rows,
        BLOCK_HEADS=block_heads,
        BLOCK_TOKENS=block_tokens,
    )
    return g, beta, y_q, y_k, initial_state


def gating_l2norm_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-batch gates and decode-only Q/K norm in one launch.

    Q/K are [1, decode_tokens, key_heads, head_dim] views with contiguous
    features and arbitrary token/head strides; their outputs are contiguous.
    A/B cover the full batch as [tokens, value_heads]. Inputs are not modified.
    """
    if q is None or k is None:
        raise ValueError("gating_l2norm_qk: decode Q/K are required")
    g, beta, y_q, y_k, _ = _gating_l2norm_qk(q, k, A_log, a, b, dt_bias, eps)
    return g, beta, y_q, y_k


def gating_gather_clear_decode_l2norm_qk(
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    q: torch.Tensor | None,
    k: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fuse gates, prefill state gather/clear, and optional decode Q/K norm.

    State output preserves [prefill_requests, value_heads, Dv, Dk] and the
    cache dtype. Fresh requests produce zeros without reading the cache; valid
    indices are required for requests with history. Cache/input tensors are
    never modified. Q/K must contain ONLY decode rows, or both be None for
    pure prefill: prefill Q/K normalization remains inside chunk GDN.
    """
    if ssm_state is None:
        raise ValueError("gating_l2norm_qk: prefill state is required")
    g, beta, y_q, y_k, initial_state = _gating_l2norm_qk(
        q, k, A_log, a, b, dt_bias, eps, ssm_state, state_indices, has_initial_state
    )
    return g, beta, initial_state, y_q, y_k

