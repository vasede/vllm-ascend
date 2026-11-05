# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Non-speculative GDN decode using xLLM's MegaGdnDecode ACLNN operator."""

import math
from importlib import import_module

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.utils import get_weight_prefetch_method, is_950

HEAD_DIM = 128
CONV_HISTORY = 3
MAX_BATCH_SIZE = 32
MAX_CACHE_SLOTS = 1024
STATE_IO_BLOCK = 1024
NORM_EPS = 1e-6
logger = init_logger(__name__)


def mega_gdn_decode_enabled() -> bool:
    # Configuration is immutable for a worker's lifetime.
    from vllm_ascend.ascend_config import get_ascend_config

    try:
        config = get_ascend_config()
    except RuntimeError:
        # Standalone layer utilities can run before worker configuration.
        return False
    return config.ascend_fusion_config.fusion_ops_mega_gdn_decode and is_950()


@torch.no_grad()
def prepare_mega_gdn_weights(layer: torch.nn.Module) -> None:
    """Refresh the FP32 bias after loading, including layerwise reload."""
    if not hasattr(layer, "dt_bias") or layer.dt_bias.is_meta:
        return
    if layer.dt_bias.device.type == "npu":
        # Load only after device selection, before any graph capture.
        import_module("vllm_ascend.vllm_ascend_C")
        if not hasattr(torch.ops._C_ascend, "mega_gdn_decode"):
            raise RuntimeError("Rebuild vllm_ascend_C to enable MegaGdnDecode")
        layer._mega_gdn_strided_ready = (
            hasattr(torch.ops._C_ascend, "has_mega_gdn_decode_strided")
            and torch.ops._C_ascend.has_mega_gdn_decode_strided()
        )
        layer._mega_gdn_strided_batch_ready = (
            hasattr(torch.ops._C_ascend, "has_mega_gdn_decode_strided_batch")
            and torch.ops._C_ascend.has_mega_gdn_decode_strided_batch()
        )
    source = layer.dt_bias.detach().float().contiguous()
    name = "_mega_gdn_dt_bias"
    current = getattr(layer, name, None)
    if current is not None and current.shape == source.shape and current.device == source.device:
        current.copy_(source)
    else:
        layer.register_buffer(name, source.clone(), persistent=False)


def supports_mega_gdn_decode(layer, metadata, qkv, z, b, a, conv_weight, output) -> bool:
    """Only inspect host metadata; never synchronize device indices."""
    if (
        layer.num_spec != 0
        or metadata.num_prefills != 0
        or metadata.num_spec_decodes != 0
        or metadata.spec_sequence_masks is not None
        or metadata.num_decodes <= 0
        or layer.head_k_dim != HEAD_DIM
        or layer.head_v_dim != HEAD_DIM
        or layer.conv_kernel_size != CONV_HISTORY + 1
        or layer.conv1d.bias is not None
        or layer.activation not in ("silu", "swish")
        or layer.norm.eps != NORM_EPS
        or not layer.norm.norm_before_gate
        or layer.norm.group_size is not None
        or getattr(layer.norm, "activation", "silu") not in ("silu", "swish")
    ):
        return False
    batch, channels = qkv.shape
    nk = layer.num_k_heads // layer.tp_size
    nv = layer.num_v_heads // layer.tp_size
    if (
        not 1 <= batch <= MAX_BATCH_SIZE
        or nk not in (1, 2, 4, 8, 16)
        or nv % nk != 0
        or not 1 <= nv // nk <= 4
        or channels != (2 * nk + nv) * HEAD_DIM
        or z.shape != (batch, nv, HEAD_DIM)
        or b.shape != (batch, nv)
        or a.shape != b.shape
        or conv_weight.shape != (CONV_HISTORY + 1, channels)
        or layer.norm.weight.shape != (HEAD_DIM,)
        or layer.A_log.shape != (nv,)
    ):
        return False
    conv, ssm = layer.kv_cache
    indices = metadata.non_spec_state_indices_tensor
    bias = getattr(layer, "_mega_gdn_dt_bias", None)
    if (
        conv.ndim != 3
        or conv.shape[1:] != (CONV_HISTORY, channels)
        or ssm.shape != (conv.shape[0], nv, HEAD_DIM, HEAD_DIM)
        or conv.shape[0] == 0
        or conv.dtype != torch.bfloat16
        or ssm.dtype != torch.float32
        or layer.A_log.dtype != torch.float32
        or bias is None
        or bias.shape != (nv,)
        or bias.dtype != torch.float32
        or indices is None
        or indices.ndim != 1
        or indices.numel() < batch
        or indices.dtype not in (torch.int32, torch.int64)
        or output.shape[1:] != z.shape[1:]
        or output.shape[0] < batch
        or not output.is_contiguous()
    ):
        return False
    tensors = (qkv, z, b, a, conv_weight, layer.norm.weight, output)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        return False
    if any(t.device != qkv.device for t in (*tensors, conv, ssm, indices, bias, layer.A_log)):
        return False
    # Padded physical pages are supported; inner row layouts must be dense.
    return all(t[0].is_contiguous() and t.stride(0) >= math.prod(t.shape[1:]) for t in (conv, ssm))


@triton.jit
def _copy_decode_states(
    conv,
    ssm,
    indices,
    compact_conv,
    compact_ssm,
    compact_indices,
    conv_stride: tl.int64,
    ssm_stride: tl.int64,
    indices_stride,
    slots,
    conv_size: tl.constexpr,
    ssm_size: tl.constexpr,
    STORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(1)
    tile = tl.program_id(0)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(indices + request * indices_stride).to(tl.int64)
    # Block zero is the model runner's dedicated graph-padding slot.
    live = (slot > 0) & (slot < slots)
    conv_mask = offset < conv_size
    ssm_mask = offset < ssm_size
    if STORE:
        cv = tl.load(compact_conv + request * conv_size + offset, conv_mask, other=0)
        sv = tl.load(compact_ssm + request * ssm_size + offset, ssm_mask, other=0)
        tl.store(conv + slot * conv_stride + offset, cv, conv_mask & live)
        tl.store(ssm + slot * ssm_stride + offset, sv, ssm_mask & live)
    else:
        cv = tl.load(conv + slot * conv_stride + offset, conv_mask & live, other=0)
        sv = tl.load(ssm + slot * ssm_stride + offset, ssm_mask & live, other=0)
        tl.store(compact_conv + request * conv_size + offset, cv, conv_mask)
        tl.store(compact_ssm + request * ssm_size + offset, sv, ssm_mask)
        if tile == 0:
            tl.store(compact_indices + request, request)


def run_mega_gdn_decode(layer, metadata, qkv, z, b, a, conv_weight, output) -> None:
    """Run after eligibility checks and KV connector waits; mutate live caches."""
    conv, ssm = layer.kv_cache
    batch = qkv.shape[0]
    indices = metadata.non_spec_state_indices_tensor[:batch]
    native_op = None
    if 1 <= batch <= MAX_BATCH_SIZE and getattr(layer, "_mega_gdn_strided_batch_ready", False):
        logger.info_once("Using MegaGdnDecodeStridedBatch for B1-32 without state gather/scatter.")
        native_op = torch.ops._C_ascend.mega_gdn_decode_strided_batch
    elif batch == 1 and getattr(layer, "_mega_gdn_strided_ready", False):
        logger.info_once("Using MegaGdnDecodeStrided for B1 without state gather/scatter.")
        native_op = torch.ops._C_ascend.mega_gdn_decode_strided
    if native_op is not None:
        index_arg = indices.to(torch.int32).contiguous()
        conv_out = torch.empty_like(qkv, memory_format=torch.contiguous_format)
        native_op(
            qkv.contiguous(),
            z.contiguous(),
            b.contiguous(),
            a.contiguous(),
            conv_weight.contiguous(),
            conv,
            layer.A_log.contiguous(),
            layer._mega_gdn_dt_bias,
            ssm,
            index_arg,
            index_arg,
            layer.norm.weight.contiguous(),
            conv_out,
            output[:batch],
            False,
        )
        return
    compact = not (conv.is_contiguous() and ssm.is_contiguous() and conv.shape[0] <= MAX_CACHE_SLOTS)
    logger.info_once("Using MegaGdnDecode for non-speculative pure decode (compact state cache: %s).", compact)
    if compact:
        conv_arg = torch.empty((batch, *conv.shape[1:]), dtype=conv.dtype, device=conv.device)
        ssm_arg = torch.empty((batch, *ssm.shape[1:]), dtype=ssm.dtype, device=ssm.device)
        index_arg = torch.empty(batch, dtype=torch.int32, device=qkv.device)
        conv_size, ssm_size = math.prod(conv.shape[1:]), math.prod(ssm.shape[1:])
        grid = (triton.cdiv(max(conv_size, ssm_size), STATE_IO_BLOCK), batch)
        _copy_decode_states[grid](
            conv,
            ssm,
            indices,
            conv_arg,
            ssm_arg,
            index_arg,
            conv.stride(0),
            ssm.stride(0),
            indices.stride(0),
            conv.shape[0],
            conv_size,
            ssm_size,
            STORE=False,
            BLOCK=STATE_IO_BLOCK,
        )
    else:
        conv_arg, ssm_arg = conv, ssm
        index_arg = indices.to(torch.int32).contiguous()
    conv_out = torch.empty_like(qkv, memory_format=torch.contiguous_format)
    torch.ops._C_ascend.mega_gdn_decode(
        qkv.contiguous(),
        z.contiguous(),
        b.contiguous(),
        a.contiguous(),
        conv_weight.contiguous(),
        conv_arg,
        layer.A_log.contiguous(),
        layer._mega_gdn_dt_bias,
        ssm_arg,
        index_arg,
        index_arg,
        layer.norm.weight.contiguous(),
        conv_out,
        output[:batch],
        False,
    )
    if compact:
        _copy_decode_states[grid](
            conv,
            ssm,
            indices,
            conv_arg,
            ssm_arg,
            index_arg,
            conv.stride(0),
            ssm.stride(0),
            indices.stride(0),
            conv.shape[0],
            conv_size,
            ssm_size,
            STORE=True,
            BLOCK=STATE_IO_BLOCK,
        )


def ascend_gdn_core_with_norm(
    qkv: torch.Tensor, z: torch.Tensor, b: torch.Tensor, a: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    fused = layer._forward_core(qkv, b, a, output, mega_z=z)
    method = get_weight_prefetch_method()
    ratio = float(method.attn.prefetch_ratio.get("o", 0)) if method is not None and method.attn.enable else 0.0
    if fused:
        # The usual gated-norm op owns this join; mega already did the norm.
        from vllm_ascend.ops.mlp_weight_prefetch import MLP_PREFETCH_TOKEN_THRESHOLD, _finish_prefetch, _prefetch_size
        from vllm_ascend.utils import prefetch_stream

        if (
            ratio > 0
            and _prefetch_size(layer.out_proj.weight, ratio) > 0
            and 0 < qkv.shape[0] < MLP_PREFETCH_TOKEN_THRESHOLD
        ):
            _finish_prefetch(prefetch_stream())
    else:
        flat = output.reshape(-1, output.shape[-1])
        z_flat = z.reshape(-1, z.shape[-1])
        if ratio > 0:
            result = torch.ops.vllm.prefetch_gated_norm(
                flat,
                z_flat,
                layer.norm.weight,
                layer.out_proj.weight,
                layer.norm.eps,
                -1 if layer.norm.group_size is None else layer.norm.group_size,
                layer.norm.norm_before_gate,
                ratio,
                qkv.shape[0],
            )
        else:
            result = layer.norm(flat, z_flat)
        output.copy_(result.reshape_as(output))


def _ascend_gdn_core_with_norm_fake(
    qkv: torch.Tensor, z: torch.Tensor, b: torch.Tensor, a: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None:
    return None


direct_register_custom_op(
    op_name="ascend_gdn_core_with_norm",
    op_func=ascend_gdn_core_with_norm,
    mutates_args=["output"],
    fake_impl=_ascend_gdn_core_with_norm_fake,
)
