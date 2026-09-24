#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from collections.abc import Iterable

import torch
import torch_npu
from einops import rearrange
from vllm.logger import init_logger
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.triton_utils import triton
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import fused_qkvzba_split_reshape_cat
from vllm_ascend.ops.triton.fla.utils import (
    clear_ssm_states,
    gating_gather_clear_l2norm_qk,
    preamble_fusion_enabled,
)
from vllm_ascend.ops.triton.mamba.causal_conv1d import extract_last_width
from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update

logger = init_logger(__name__)

# Importing fla_npu is what puts the custom operator package on ASCEND_CUSTOM_OPP_PATH
# (and its libcust_opapi.so on LD_LIBRARY_PATH). CANN reads that path when it loads the
# kernel registry at device init, so an import that happens later -- inside the function,
# say -- leaves the kernel unresolvable and every shape fails with aclnnStatus=169112.
# ascend_config imports this eagerly when gdn_fused_chunk_op="fla_npu" for the same reason;
# this import is the fallback for direct users of the function below.
try:
    import fla_npu  # noqa: F401
    
    from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_fwd
    from fla_npu.ops.ascendc import npu_recurrent_gated_delta_rule
except ImportError:
    npu_chunk_gated_delta_rule_fwd = None


def _chunk_gated_delta_rule_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    actual_seq_lengths: torch.Tensor | None = None,
    qk_normalized: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused prefill path using ``torch_npu.npu_chunk_gated_delta_rule``.

    Drop-in replacement for the Triton ``chunk_gated_delta_rule`` pipeline
    (chunk_scaled_dot_kkt_fwd + solve_tril + recompute_w_u_fwd + ...).
    The fused CANN operator expects TND layout and does NOT apply q/k L2 norm
    or the chunk-local cumsum of ``g`` internally, so q/k are normalized here
    and the raw ``g`` is passed through.

    Args:
        q, k: ``[1, T, Nk, Dk]``   v: ``[1, T, Nv, Dv]``
        g, beta: ``[1, T, Nv]``    g is fp32 (<=0), beta is (0, 1).
        initial_state: ``[N, Nv, Dv, Dk]`` — same layout as ``ssm_state``,
            no transpose required.
        cu_seqlens: cumulative prefill query start locations ``[N+1]``.
        scale: query scaling factor (``Dk ** -0.5``).

    Returns:
        o: ``[1, T, Nv, Dv]`` and final_state: ``[N, Nv, Dv, Dk]``.
    """
    # TND layout: drop the leading batch dim (batch size is always 1 here).
    # On the fused-preamble path gating_gather_clear_l2norm_qk already normalized q/k
    # inside its single launch, so normalizing again would be both wrong and an extra
    # launch. Every other path still needs it here.
    if not qk_normalized:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
    # Guarding these with dim/dtype checks was tried and measured at 1.0us/layer, so
    # it is not worth the extra branching: PyTorch already short-circuits squeeze on a
    # non-unit dim, contiguous() on contiguous input and to() on a matching dtype in
    # C++ before any dispatch. (The ~1.4us/op seen in profiles is observer effect.)
    q = q.squeeze(0).contiguous()  # [T, Nk, Dk]
    k = k.squeeze(0).contiguous()  # [T, Nk, Dk]
    v = v.squeeze(0).contiguous()  # [T, Nv, Dv]
    g = g.squeeze(0).to(torch.float32).contiguous()  # [T, Nv]
    beta = beta.squeeze(0).to(v.dtype).contiguous()  # [T, Nv]

    # The fused op only supports a bfloat16 initial_state, while ssm_state may be
    # float32 (the recurrent path keeps fp32 state). gating_gather_clear_l2norm_qk already
    # gathers straight into a contiguous bf16 buffer, so both calls short-circuit on
    # the fused path; they only do work for callers that pass an fp32 state.
    initial_state = initial_state.to(torch.bfloat16).contiguous()

    # actual_seq_lengths is per-batch sequence length [N] (per the interface doc),
    # derived from the cumulative query_start_loc. The builder precomputes it once
    # per step; deriving it here would issue one redundant launch per GDN layer.
    if actual_seq_lengths is None:
        actual_seq_lengths = torch.diff(cu_seqlens).to(torch.int32)

    o, final_state = torch_npu.npu_chunk_gated_delta_rule(
        q,
        k,
        v,
        beta=beta,
        initial_state=initial_state,
        actual_seq_lengths=actual_seq_lengths,
        scale=scale,
        g=g,
    )
    return o.unsqueeze(0), final_state


def _resolve_host_cu_seqlens_and_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    host_cu_seqlens: tuple[int, ...] | None,
    host_chunk_indices: tuple[int, ...] | None,
    host_chunk_size: int | None,
) -> tuple[list[int], list[int]]:
    """Host-side cu_seqlens/chunk_indices for the Phase 6 aclIntArray parameters.

    Uses the builder's precomputed copies when they apply, which is what keeps the
    synchronous ``cu_seqlens.tolist()`` off the per-layer path. Falls back to deriving
    them here whenever the cache cannot be trusted: a different ``chunk_size`` than the
    builder assumed, or a caller that passes no metadata at all (the unit tests and any
    direct user of this function).
    """
    if (
        host_cu_seqlens is not None
        and host_chunk_indices is not None
        # The builder precomputes for one chunk_size only; a caller asking for a
        # different chunking would otherwise get indices for the wrong chunk count.
        and host_chunk_size == chunk_size
        # Cheap guard against a stale cache. The kernel validates cu_seqlens[-1] == T
        # and the canonical order anyway, but catching a length mismatch here keeps the
        # failure readable instead of surfacing as an aclnn error.
        and len(host_cu_seqlens) == cu_seqlens.numel()
    ):
        return list(host_cu_seqlens), list(host_chunk_indices)

    cu_seqlens_list = [int(x) for x in cu_seqlens.tolist()]
    chunk_indices_list: list[int] = []
    # chunk_indices must be canonical sequence-major (seq_idx, local_chunk) pairs,
    # flattened -- the kernel re-derives and compares them, and raises otherwise.
    for seq_idx in range(len(cu_seqlens_list) - 1):
        seq_len = cu_seqlens_list[seq_idx + 1] - cu_seqlens_list[seq_idx]
        if seq_len <= 0:
            continue
        for local_chunk in range((seq_len + chunk_size - 1) // chunk_size):
            chunk_indices_list.extend((seq_idx, local_chunk))
    return cu_seqlens_list, chunk_indices_list


def _chunk_gated_delta_rule_fla_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    actual_seq_lengths: torch.Tensor | None = None,
    qk_normalized: bool = False,
    chunk_size: int = 64,
    host_cu_seqlens: tuple[int, ...] | None = None,
    host_chunk_indices: tuple[int, ...] | None = None,
    host_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused prefill path using the flash-linear-attention-npu Phase 6 kernel.

    Same contract as :func:`_chunk_gated_delta_rule_fused` so both can sit behind
    the ``gdn_fused_chunk_op`` switch, but dispatches to
    ``fla_npu.ops.ascendc.npu_chunk_gated_delta_rule_fwd``
    (``aclnnChunkGatedDeltaRuleFwd``, the single-kernel fused entry) instead of the
    built-in CANN op.

    Modelled on ``flash_chunk_gated_delta_rule_fwd(use_composite_core=True)`` in
    the upstream ``examples/flash_gated_delta_rule.py``: the kernel folds
    chunk-local cumsum of ``g``, KKT, solve_tri, recompute_w_u, fwd_h and fwd_o
    into one launch, so ``g`` is passed raw. It does NOT apply the q/k L2 norm,
    which is therefore still done here.

    Layout differences from the CANN op, which is most of what this wrapper does:
      * Phase 6 wants BNSD (``[B, N, S, D]``), not TND, and keeps the batch dim.
      * It supports GVA natively (``Nv % Nk == 0``), so q/k must NOT be expanded
        to ``Nv`` heads the way the earlier phase checkpoints require.
      * State is ``[N, Nv, Dk, Dv]``, the transpose of ``ssm_state``.
      * ``cu_seqlens``/``chunk_indices`` are plain int lists, not tensors.

    Args:
        q, k: ``[1, T, Nk, Dk]``   v: ``[1, T, Nv, Dv]``
        g, beta: ``[1, T, Nv]``    g is fp32 (<=0), beta is (0, 1).
        initial_state: ``[N, Nv, Dv, Dk]`` — ``ssm_state`` layout.
        cu_seqlens: cumulative prefill query start locations ``[N+1]``.
        scale: query scaling factor (``Dk ** -0.5``).
        actual_seq_lengths: unused, accepted for signature parity with the CANN path.
        chunk_size: kernel chunk size, must be 64 or 128.

    Returns:
        o: ``[1, T, Nv, Dv]`` and final_state: ``[N, Nv, Dv, Dk]``.
    """
    # if not qk_normalized:
    #     q = l2norm_fwd(q)
    #     k = l2norm_fwd(k)

    if npu_chunk_gated_delta_rule_fwd is None: 
        raise RuntimeError(
            "gdn_fused_chunk_op='fla_npu' needs the flash-linear-attention-npu package, "
            "which failed to import. Install it, or use gdn_fused_chunk_op='cann'."
        )

    if q.dim() == 3:
        q = q.unsqueeze(0)
    if k.dim() == 3:
        k = k.unsqueeze(0)
    if v.dim() == 3:
        v = v.unsqueeze(0)
    
    num_v_heads, head_v_dim = v.shape[1], v.shape[3]
    num_k_heads, head_k_dim = k.shape[1], k.shape[3]
    # Fail loudly instead of letting the aclnn shape check raise a message that
    # gives no hint about which model config walked into an unsupported kernel.
    if head_k_dim != 128 or head_v_dim not in (128, 256):
        raise ValueError(
            f"gdn_fused_chunk_op='fla_npu' requires head_k_dim=128 and head_v_dim in (128, 256), "
            f"got head_k_dim={head_k_dim}, head_v_dim={head_v_dim}."
        )
    if num_v_heads % num_k_heads != 0:
        raise ValueError(
            f"gdn_fused_chunk_op='fla_npu' requires num_v_heads divisible by num_k_heads, "
            f"got num_v_heads={num_v_heads}, num_k_heads={num_k_heads}."
        )

    # BSND -> BNSD. Phase 6 consumes q/k at Nk heads and expands to Nv internally.
    # q = q.transpose(1, 2).contiguous()  # [1, Nk, T, Dk]
    # k = k.transpose(1, 2).contiguous()  # [1, Nk, T, Dk]
    # v = v.transpose(1, 2).contiguous()  # [1, Nv, T, Dv]
    # g/beta are already [B, T, Nv], which is the layout the kernel wants. g stays
    # fp32 and un-cumsummed; the kernel does the chunk-local cumsum itself.
    # g = g.to(torch.float32).contiguous()
    # beta = beta.to(v.dtype).contiguous()

    # ssm_state is [N, Nv, Dv, Dk] but the kernel state is [N, Nv, Dk, Dv].
    # fp32 state is not supported, and final_state comes back in the dtype of
    # initial_state, so cast here and let the caller cast back on write-back.
    # initial_state = initial_state.transpose(-1, -2).to(v.dtype).contiguous()

    # The aclnn entry takes cu_seqlens/chunk_indices as host int arrays. Deriving
    # them here means calling .tolist() on a device tensor, which is a *synchronous*
    # D2H: it drains the stream, so the device sits idle while the host finishes the
    # launch. Profiling a single-request 11k-token prefill measured that bubble at
    # 614us per GDN layer -- 93ms over 30 layers, 15.6% of the kernel time itself --
    # of which only 27us was the Python list building. The builder already holds the
    # same values on the host (it is handed prefill_query_start_loc_cpu), so prefer
    # its copy and keep the local derivation only as a fallback.
    cu_seqlens_list, chunk_indices_list = _resolve_host_cu_seqlens_and_chunk_indices(
        cu_seqlens, chunk_size, host_cu_seqlens, host_chunk_indices, host_chunk_size
    )

    # backward pass, which prefill never runs, but they cannot be skipped: passing
    # disable_recompute=True drops them from the output tuple and the aclnn entry then
    # rejects the call with aclnnStatus=169104, so keep the default and discard them.
    o, final_state, _, _, _, _, _, _, _, _ = npu_chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        use_exp2=True,
        use_qk_l2norm_in_kernel=True,
        disable_recompute=False,
        state_v_first=True,
        output_final_state=True,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list,
        scale=scale,
        layout="BNSD",
    )

    return o, final_state


def _rearrange_decode_qkv(
    mixed_qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return contiguous decode Q/K/V without concatenating them again.

    The recurrent kernel and l2norm consume contiguous token-major tensors.
    Each component is copied separately only when its split view has gaps
    between tokens. A single-token input needs no data copy.
    """
    num_tokens = mixed_qkv.shape[0]
    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    return (
        query.contiguous().view(1, num_tokens, -1, head_k_dim),
        key.contiguous().view(1, num_tokens, -1, head_k_dim),
        value.contiguous().view(1, num_tokens, -1, head_v_dim),
    )


def _cached_conv_weights_t(layer) -> torch.Tensor:
    """Materialise conv1d weight^T once per layer instead of once per step.

    The conv1d weight is constant after loading but the aclnn op wants it
    transposed, and a plain .transpose(0, 1) is non-contiguous, so every call
    used to pay a Contiguous_Transpose kernel. Cache the contiguous result on
    the layer, re-deriving it only if the weight storage is swapped out.

    This is a module-level helper on purpose: the Qwen3.5 GDN class is built by
    copying selected methods onto the upstream class (see
    patch/worker/patch_qwen3_5.py), so a new method defined here would not be
    carried over.
    """
    w = layer.conv1d.weight
    key = (w.data_ptr(), tuple(w.shape))
    cached = getattr(layer, "_conv_weights_t_cache", None)
    if cached is not None and getattr(layer, "_conv_weights_t_src", None) == key:
        return cached
    cached = w.view(w.size(0), w.size(2)).transpose(0, 1).contiguous()
    layer._conv_weights_t_cache = cached
    layer._conv_weights_t_src = key
    return cached


class AscendGatedDeltaNetAttention(GatedDeltaNetAttention):
    def _split_ba_for_tp(self, ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "split_ba"):
            return self.split_ba(ba)
        return ba.chunk(2, dim=-1)

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
        return

    def _warmup_prefill_kernels_v0202(self, mixed_qkv: torch.Tensor) -> None:
        return

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendGDNAttentionBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        if hasattr(self, "in_proj_qkv"):
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            # FlashComm1 gathers the sequence before the input projection.
            num_tokens = mixed_qkv.size(0)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = self._split_ba_for_tp(ba)
            b = b.contiguous()
            a = a.contiguous()
        else:
            if not self.gqa_interleaved_layout:
                mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
                num_tokens = mixed_qkvz.size(0)
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                ba, _ = self.in_proj_ba(hidden_states)
                b, a = self._split_ba_for_tp(ba)

                b = b.contiguous()
                a = a.contiguous()
            else:
                projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
                projected_states_ba, _ = self.in_proj_ba(hidden_states)
                num_tokens = projected_states_qkvz.size(0)

                mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                    projected_states_qkvz,
                    projected_states_ba,
                    triton.cdiv(self.num_k_heads, self.tp_size),
                    triton.cdiv(self.num_v_heads, self.tp_size),
                    self.head_k_dim,
                    self.head_v_dim,
                )

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
            False,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        maybe_save_kv_layer_to_connector("", [])
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation (called by custom op).
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv
        # Set when a mixed batch defers its decode conv1d kernel to section 2.3,
        # so that it is issued right before the decode GDN op instead of next to
        # the prefill conv1d. Both stay None on every other path.
        decode_conv_input = None
        decode_conv_state_indices = None

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            conv_weights_T = _cached_conv_weights_t(self)
            # activation_num = 1 if self.activation else 0
            spec_causal_conv1d_meta = attn_metadata.spec_decode_metadata.spec_causal_conv1d
            spec_query_start_loc_device = spec_causal_conv1d_meta.query_start_loc

            # output_spec = torch.empty_like(mixed_qkv_spec)
            if spec_causal_conv1d_meta.cache_indices.dim() == 1:
                conv_state_indices=spec_causal_conv1d_meta.cache_indices.contiguous()
            elif spec_causal_conv1d_meta.cache_indices.dim() == 2:
                conv_state_indices=spec_causal_conv1d_meta.cache_indices[:, 0].contiguous()
           
            # See the out= note on the decode call below: without it the op ends
            # in a dead `x.copy_(result)`, which is a ViewCopy kernel here because
            # in the steady MTP state mixed_qkv_spec aliases mixed_qkv.
            output_spec = causal_conv1d_update(
                mixed_qkv_spec,
                weight=conv_weights_T,
                conv_state=self_kv_cache[0],
                bias=self.conv1d.bias,
                query_start_loc=spec_query_start_loc_device,
                conv_state_indices=spec_causal_conv1d_meta.cache_indices[:, 0].contiguous(),
                num_accepted_tokens=spec_causal_conv1d_meta.num_accepted_tokens,
                activation=self.activation,
                out=torch.empty_like(mixed_qkv_spec),
                # pad_slot_id=PAD_SLOT_ID,
                max_query_len=spec_state_indices_tensor.size(-1),
            )
            mixed_qkv_spec = output_spec
            query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                non_spec_causal_conv1d_meta = attn_metadata.non_spec_prefill_metadata.causal_conv1d
                query_start_loc_opt = non_spec_causal_conv1d_meta.query_start_loc
                cache_indices_opt = non_spec_causal_conv1d_meta.cache_indices
                initial_state_mode_opt = non_spec_causal_conv1d_meta.initial_state_mode
                if get_pcp_group().world_size > 1:
                    conv_weights_T = _cached_conv_weights_t(self)
                    # activation_num = 1 if self.activation else 0
                    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
                    assert non_spec_query_start_loc is not None
                    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
                    width = conv_weights.shape[1]
                    state_len = width - 1
                    num_seqs = non_spec_query_start_loc.shape[0] - 1
                    prefill_seq_offset = max(0, num_seqs - attn_metadata.num_prefills)
                    prefill_cache_indices = non_spec_state_indices_tensor[prefill_seq_offset:]
                    mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
                    last_width_prefill_x = extract_last_width(
                        mixed_qkv_non_spec_T, non_spec_query_start_loc[prefill_seq_offset:], state_len
                    )
                    pcp_rank = get_pcp_group().rank_in_group
                    all_last_width_prefill_x = get_pcp_group().all_gather(
                        last_width_prefill_x.unsqueeze(0).contiguous(), 0
                    )
                    if pcp_rank > 0 and prefill_cache_indices.shape[0] > 0:
                        self_kv_cache[0][prefill_cache_indices, :state_len, :] = all_last_width_prefill_x[
                            pcp_rank - 1, ...
                        ].transpose(-1, -2)
                    # mixed_qkv_non_spec_output = torch.empty_like(mixed_qkv_non_spec)
                    # print(f"=====================query_start_loc_opt:{query_start_loc_opt.get_device()}=================")
                    if cache_indices_opt.dim() == 1:
                        cache_indices=cache_indices_opt.contiguous()
                    elif cache_indices_opt.dim() == 2:
                        cache_indices=cache_indices_opt[:, 0].contiguous()
                    
                    mixed_qkv_non_spec_output = causal_conv1d_fn(
                        mixed_qkv_non_spec,
                        conv_weights_T,
                        conv_states=self_kv_cache[0],
                        bias=self.conv1d.bias,
                        query_start_loc=query_start_loc_opt,
                        cache_indices=cache_indices,
                        has_initial_state=initial_state_mode_opt,
                        # num_accepted_tokens_opt=None,
                        activation=self.activation,
                        pad_slot_id=PAD_SLOT_ID,
                        head_num=(self.num_k_heads+self.num_k_heads+self.num_v_heads)//self.tp_size
                        # run_mode=0,
                    )
                    mixed_qkv_non_spec = mixed_qkv_non_spec_output
                    if prefill_cache_indices.shape[0] > 0:
                        self_kv_cache[0][prefill_cache_indices, :state_len, :] = all_last_width_prefill_x[
                            -1, ...
                        ].transpose(-1, -2)
                    query_non_spec, key_non_spec, value_non_spec = torch.split(mixed_qkv_non_spec, [self.num_k_heads//self.tp_size, self.num_k_heads//self.tp_size, self.num_v_heads//self.tp_size], dim=-3)
                else:
                    conv_weights_T = _cached_conv_weights_t(self)

                    if cache_indices_opt.dim() == 1:
                        cache_indices=cache_indices_opt.contiguous()
                    elif cache_indices_opt.dim() == 2:
                        cache_indices=cache_indices_opt[:, 0].contiguous()
                    
                    # Mixed non-spec batch: drive the two conv1d kernels separately so
                    # each side already carries the layout its core-attention op wants.
                    # Decode rows lead the non-spec batch (the builder rebases prefill
                    # offsets by num_decode_tokens), so both slices stay contiguous.
                    split_conv = spec_sequence_masks is None and attn_metadata.num_decodes > 0
                    if split_conv:
                        num_decodes_conv = attn_metadata.num_decodes
                        num_decode_tokens_conv = attn_metadata.num_decode_tokens
                        decode_conv_meta = attn_metadata.non_spec_decode_metadata.causal_conv1d
                        if decode_conv_meta.cache_indices.dim() == 1:
                            decode_conv_state_indices = decode_conv_meta.cache_indices[
                                :num_decodes_conv
                            ].contiguous()
                        else:
                            decode_conv_state_indices = decode_conv_meta.cache_indices[
                                :num_decodes_conv, 0
                            ].contiguous()
                        # Preserve decode rows before prefill replaces mixed_qkv_non_spec.
                        decode_conv_input = mixed_qkv_non_spec[:num_decode_tokens_conv]
                        # Prefill-only view. prefill_query_start_loc is already rebased
                        # to 0 by the builder; the per-row metadata is sliced past the
                        # decode rows exactly like prefill_has_initial_state is.
                        prefill_conv_input = mixed_qkv_non_spec[num_decode_tokens_conv:]
                        prefill_qsl_conv = attn_metadata.prefill_query_start_loc
                        prefill_cache_indices_conv = cache_indices[num_decodes_conv:]
                        prefill_initial_state_conv = (
                            initial_state_mode_opt[num_decodes_conv:]
                            if initial_state_mode_opt is not None
                            else None
                        )
                    else:
                        prefill_conv_input = mixed_qkv_non_spec
                        prefill_qsl_conv = query_start_loc_opt
                        prefill_cache_indices_conv = cache_indices
                        prefill_initial_state_conv = initial_state_mode_opt

                    # NTD (head-first [N, T, D]) for npu_chunk_gated_delta_rule_fwd.
                    mixed_qkv_non_spec_output = causal_conv1d_fn(
                        prefill_conv_input,
                        conv_weights_T,
                        conv_states=self_kv_cache[0],
                        bias=self.conv1d.bias,
                        query_start_loc=prefill_qsl_conv,
                        cache_indices=prefill_cache_indices_conv,
                        has_initial_state=prefill_initial_state_conv,
                        activation=self.activation,
                        pad_slot_id=PAD_SLOT_ID,
                        head_num=(self.num_k_heads+self.num_k_heads+self.num_v_heads)//self.tp_size
                    )
                    mixed_qkv_non_spec = mixed_qkv_non_spec_output
                    query_non_spec, key_non_spec, value_non_spec = torch.split(mixed_qkv_non_spec, [self.num_k_heads//self.tp_size, self.num_k_heads//self.tp_size, self.num_v_heads//self.tp_size], dim=-3)

                    # The decode conv1d kernel itself is deferred to section 2.3
                    # so the issue order becomes prefill conv1d -> prefill GDN ->
                    # decode conv1d -> decode GDN. Only the inputs are captured
                    # here, because the rebind above makes mixed_qkv_non_spec
                    # prefill-only from this point on.
        elif attn_metadata.num_decodes > 0:
            conv_weights_T = _cached_conv_weights_t(self)
            # activation_num = 1 if self.activation else 0
            non_spec_causal_conv1d_meta = attn_metadata.non_spec_decode_metadata.causal_conv1d
            non_spec_query_start_loc_device = non_spec_causal_conv1d_meta.query_start_loc
            # output_non_spec = torch.empty_like(mixed_qkv_non_spec)
            # output_non_spec decode [T, N*D]  prefill [N, T, D]
            
            if non_spec_causal_conv1d_meta.cache_indices.dim() == 1:
                conv_state_indices=non_spec_causal_conv1d_meta.cache_indices.contiguous()
            elif non_spec_causal_conv1d_meta.cache_indices.dim() == 2:
                conv_state_indices=non_spec_causal_conv1d_meta.cache_indices[:, 0].contiguous()
            
            # Pass an explicit out= buffer. Without it the fla_npu stable path
            # ends in `x.copy_(result); return x` to honour its "mutates x"
            # contract, and since x is a view into mixed_qkv that copy shows up
            # as a ViewCopy kernel per layer per step. We only consume the
            # return value, so that write-back is dead: handing the op its own
            # output buffer takes the early-return branch and drops the kernel.
            output_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                weight=conv_weights_T,
                conv_state=self_kv_cache[0],
                bias=self.conv1d.bias,
                # query_start_loc=non_spec_query_start_loc_device,
                conv_state_indices=conv_state_indices,
                # initial_state_mode_opt=None,
                num_accepted_tokens=None,
                activation=self.activation,
                out=torch.empty_like(mixed_qkv_non_spec),
                # max_query_len=1,
                # pad_slot_id=PAD_SLOT_ID,
                # run_mode=1,
            )
            mixed_qkv_non_spec = output_non_spec
            
            query_non_spec, key_non_spec, value_non_spec = _rearrange_decode_qkv(
                mixed_qkv_non_spec,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            mixed_qkv_non_spec = None


        # 2. Recurrent attention
        split_non_spec = (
            spec_sequence_masks is None and attn_metadata.num_prefills > 0 and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens

        # Select a fused operator via env var, then pick which one with
        # gdn_fused_chunk_op ("cann" or "fla_npu"). Both fused ops only support the
        # non-PCP case; keep the Triton pipeline as default.
        use_fused_chunk = get_ascend_config().enable_gdn_fused_chunk and get_pcp_group().world_size == 1
        # The whole preamble - gating, the ssm gather, the clear, the bf16 cast and
        # both l2norms - collapses into one Triton launch. Gating is elementwise over
        # tokens so it covers the full batch, while the gather/clear/l2norm segments
        # cover only the prefill slice; the kernel sizes those independently
        # (NUM_BATCHES vs M), so a mixed decode+prefill batch is handled by giving it
        # full-range a/b and prefill-sliced q/k. Spec-decode still needs the
        # index_select path below, and VLLM_ASCEND_GDN_FUSE_CLEAR_L2NORM=0 keeps the
        # original per-op path.
        fuse_preamble = (
            use_fused_chunk
            and preamble_fusion_enabled()
            and spec_sequence_masks is None
            and attn_metadata.num_prefills > 0
        )

        # Set by the fused call so the two branches below reuse its outputs.
        fused_prefill_qk = None
        fused_decode_qk = None
        fused_initial_state = None
        fuse_preamble = False
        if fuse_preamble:
            # Stale w.r.t. the split conv1d path: q/k below are prefill-only now,
            # while the y_q/y_k slicing further down still assumes they span the
            # whole batch. Re-enabling this for a mixed batch would double-slice.
            assert not split_non_spec, (
                "fused GDN preamble needs full-batch q/k; the split conv1d path "
                "hands it prefill-only tensors"
            )
            # q/k cover the WHOLE batch: l2norm is a per-token row reduction over the
            # feature dim, so normalizing all rows and slicing is bit-identical to
            # normalizing each slice on its own. That lets the decode branch reuse
            # these outputs instead of issuing two more l2norm launches (measured at
            # 291us of host time each on the eager path).
            g, beta, fused_initial_state, y_q, y_k = gating_gather_clear_l2norm_qk(
                ssm_state,
                attn_metadata.prefill_state_indices,
                attn_metadata.prefill_has_initial_state,
                query_non_spec,
                key_non_spec,
                self.A_log,
                a,
                b,
                self.dt_bias,
                out_dtype=torch.bfloat16,
            )
            # Slicing dim 1 of a [1, T, H, D] contiguous tensor stays contiguous
            # because dim 0 has size 1, so both slices are still flat-indexable.
            if split_non_spec:
                fused_decode_qk = (y_q[:, :num_decode_tokens], y_k[:, :num_decode_tokens])
                fused_prefill_qk = (y_q[:, num_decode_tokens:], y_k[:, num_decode_tokens:])
            else:
                fused_prefill_qk = (y_q, y_k)
        else:
            g, beta = DeviceOperator.fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            actual_seq_lengths = attn_metadata.spec_decode_metadata.actual_seq_lengths
            query_spec = l2norm_fwd(query_spec)
            key_spec = l2norm_fwd(key_spec)
            # Dispatches to the vllm-ascend AscendC custom operator
            # (csrc/recurrent_gated_delta_rule), NOT the built-in CANN operator.
            # The custom op extends dtype support (e.g. float32 state) and is
            # loaded at runtime via ASCEND_CUSTOM_OPP_PATH.
            core_attn_out_spec = npu_recurrent_gated_delta_rule(
                query=query_spec.squeeze(0),
                key=key_spec.squeeze(0),
                value=value_spec.squeeze(0),
                g=g_spec.squeeze(0),
                beta=beta_spec.squeeze(0),
                state=ssm_state,
                scale=key_spec.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=spec_state_indices_tensor.flatten(),
                num_accepted_tokens=spec_causal_conv1d_meta.num_accepted_tokens.to(torch.int32),
            ).unsqueeze(0)
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Stage the non-spec-decode gating slices. Both the decode conv1d and
        # the decode GDN op run in 2.3, after the prefill GDN; the slices have to
        # be taken here because 2.3 rebinds g_non_spec/beta_non_spec to their
        # prefill-only views before it gets there.
        if split_non_spec:
            assert g_non_spec is not None
            assert beta_non_spec is not None
            decode_actual_seq_lengths = attn_metadata.non_spec_decode_metadata.actual_seq_lengths
            # None here means the conv section never captured the decode rows,
            # i.e. the PCP branch, which does not implement the split path.
            assert decode_conv_input is not None
            assert decode_conv_state_indices is not None
            g_decode = g_non_spec[:, :num_decode_tokens]
            beta_decode = beta_non_spec[:, :num_decode_tokens]
        core_attn_out_decode = None

        # 2.3: Process the remaining part
        if attn_metadata.num_prefills > 0:
            prefill_query_start_loc = attn_metadata.prefill_query_start_loc
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_query_start_loc is not None
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            assert g_non_spec is not None
            assert beta_non_spec is not None
            if split_non_spec:
                # q/k/v already cover prefill rows only: the prefill conv1d kernel
                # above was fed mixed_qkv_non_spec[num_decode_tokens:]. Gating still
                # runs over the whole batch, so g/beta keep their slice.
                g_non_spec = g_non_spec[:, num_decode_tokens:]
                beta_non_spec = beta_non_spec[:, num_decode_tokens:]
            if fused_prefill_qk is not None:
                query_non_spec, key_non_spec = fused_prefill_qk

            if use_fused_chunk:
                # The fused op's state layout [N, Nv, Dv, Dk] matches ssm_state
                # directly, so no transpose is needed. The gather gives us a fresh
                # buffer, already bf16 for the fused op, so the Cast/contiguous
                # below are no-ops.
                if fused_initial_state is not None:
                    # Gathered, cleared and cast to bf16 by the fused kernel above.
                    initial_state = fused_initial_state
                else:
                    # Advanced indexing already returns a copy, safe to clear in place.
                    initial_state = ssm_state[prefill_state_indices]
                    clear_ssm_states(initial_state, prefill_has_initial_state)
                use_fla_npu = get_ascend_config().gdn_fused_chunk_op == "fla_npu"
                fused_chunk_fn = (
                    _chunk_gated_delta_rule_fla_npu if use_fla_npu else _chunk_gated_delta_rule_fused
                )
                chunk_meta = attn_metadata.non_spec_prefill_metadata.chunk
                # Only the Phase 6 path needs the host-side copies, and only it accepts
                # these kwargs; the CANN op keeps cu_seqlens on the device.
                host_kwargs = (
                    {
                        "host_cu_seqlens": chunk_meta.cu_seqlens_host,
                        "host_chunk_indices": chunk_meta.phase6_chunk_indices_host,
                        "host_chunk_size": chunk_meta.phase6_chunk_size,
                    }
                    if use_fla_npu
                    else {}
                )
                (core_attn_out_non_spec, last_recurrent_state) = fused_chunk_fn(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    cu_seqlens=prefill_query_start_loc,
                    scale=key_non_spec.shape[-1] ** -0.5,
                    actual_seq_lengths=chunk_meta.actual_seq_lengths,
                    qk_normalized=fuse_preamble,
                    **host_kwargs,
                )
                # Kept on the aclnn path on purpose: measured on this stack a Triton
                # launch costs ~350us of host time per call while Cast + aclnnIndexPut
                # cost ~120us, so folding this write-back into a Triton kernel is a net
                # ~230us/layer loss even though it removes two launches.
                ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)
            else:
                initial_state = ssm_state[prefill_state_indices].transpose(-1, -2).contiguous()
                clear_ssm_states(initial_state, prefill_has_initial_state)
                (core_attn_out_non_spec, last_recurrent_state) = chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=prefill_query_start_loc,
                    prebuilt_meta=attn_metadata.non_spec_prefill_metadata.chunk,
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                )
                ssm_state[prefill_state_indices] = (
                    last_recurrent_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
                )
            if split_non_spec:
                # Decode conv1d runs here, after the prefill GDN, so the whole
                # decode chain (conv1d -> l2norm -> GDN) trails the prefill chain.
                # Safe against the prefill conv1d that already ran: the two write
                # disjoint conv-state slots, and decode_conv_input is a view on the
                # decode rows, which the prefill kernel never reads or writes.
                # The decode output stays token-major for the recurrent op.
                decode_conv_out = causal_conv1d_update(
                    decode_conv_input,
                    weight=conv_weights_T,
                    conv_state=self_kv_cache[0],
                    bias=self.conv1d.bias,
                    conv_state_indices=decode_conv_state_indices,
                    num_accepted_tokens=None,
                    activation=self.activation,
                )
                # Yields TND, which npu_recurrent_gated_delta_rule consumes directly.
                query_decode, key_decode, value_decode = _rearrange_decode_qkv(
                    decode_conv_out,
                    self.key_dim // self.tp_size,
                    self.value_dim // self.tp_size,
                    self.head_k_dim,
                    self.head_v_dim,
                )
                # Decode GDN op runs after the prefill one. Both update ssm_state,
                # but at disjoint rows: prefill_state_indices is
                # non_spec_state_indices_tensor[num_decodes:] (builder) while decode
                # uses [:num_decodes], so the order does not affect the result.
                # The chunk op only normalizes the prefill slice.
                query_decode = l2norm_fwd(query_decode)
                key_decode = l2norm_fwd(key_decode)
                core_attn_out_decode = npu_recurrent_gated_delta_rule(
                    query=query_decode.squeeze(0),
                    key=key_decode.squeeze(0),
                    value=value_decode.squeeze(0),
                    g=g_decode.squeeze(0),
                    beta=beta_decode.squeeze(0),
                    state=ssm_state,
                    scale=key_decode.shape[-1] ** -0.5,
                    actual_seq_lengths=decode_actual_seq_lengths,
                    ssm_state_indices=non_spec_state_indices_tensor[: attn_metadata.num_decodes],
                ).unsqueeze(0)
                # Concat order stays decode-first: it must match the batch token
                # layout, which is independent of the execution order above.
                core_attn_out_non_spec = torch.cat(
                    [core_attn_out_decode, core_attn_out_non_spec],
                    dim=1,
                )
        elif attn_metadata.num_decodes > 0:
            actual_seq_lengths = attn_metadata.non_spec_decode_metadata.actual_seq_lengths
            query_non_spec = l2norm_fwd(query_non_spec)
            key_non_spec = l2norm_fwd(key_non_spec)
            # Dispatches to the vllm-ascend AscendC custom operator
            # (csrc/recurrent_gated_delta_rule), NOT the built-in CANN operator.
            core_attn_out_non_spec = npu_recurrent_gated_delta_rule(
                query=query_non_spec.squeeze(0),
                key=key_non_spec.squeeze(0),
                value=value_non_spec.squeeze(0),
                g=g_non_spec.squeeze(0) if g_non_spec is not None else g_non_spec,
                beta=beta_non_spec.squeeze(0) if beta_non_spec is not None else beta_non_spec,
                state=ssm_state,
                scale=key_non_spec.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=non_spec_state_indices_tensor,
            ).unsqueeze(0)
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)




from vllm.model_executor.models.qwen3_5 import Qwen3_5Model as _BaseQwen3_5Model
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)

from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
)


class AscendQwen3_5Model(_BaseQwen3_5Model):

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # GDN
            # ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
            # ("in_proj_qkvz", "in_proj_z", 3),
            # self attention
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # mlp
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("in_proj_ba", "in_proj_b", 0),
            ("in_proj_ba", "in_proj_a", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        is_fused_expert = False
        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )
        fused_expert_params_mapping = [
            (f"experts.{base_layer}w13_weight", "experts.gate_up_proj", 0, "w1"),
            (f"experts.{base_layer}w2_weight", "experts.down_proj", 0, "w2"),
        ]
        num_experts = (
            self.config.num_experts if hasattr(self.config, "num_experts") else 0
        )
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        # qwen3.5 no need to transpose
                        # loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            # down_proj
                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        if success:
                            name = name_mapped
                            break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if (
                            name_mapped.endswith(".bias")
                            or name_mapped.endswith("_bias")
                        ) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

