# SPDX-License-Identifier: Apache-2.0
"""Run after sourcing the MegaGdnDecode OPP environment; one NPU is enough."""

import copy
import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.mega_gdn_decode import run_mega_gdn_decode, supports_mega_gdn_decode

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")


def _case(batch=4, nk=2, nv=4, padded=False):
    torch.manual_seed(81)
    torch.npu.set_device(0)
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    slots, channels = batch + 3, (2 * nk + nv) * 128

    def rand(shape, scale=0.2, dtype=torch.bfloat16):
        return (torch.randn(shape) * scale).to(dtype).npu()

    conv_initial = rand((slots, 3, channels))
    ssm_initial = rand((slots, nv, 128, 128), 0.05, torch.float32)

    def cache(value):
        if not padded:
            return value.clone()
        row = value[0].numel()
        storage = torch.full((slots, row + 256), 9.0, dtype=value.dtype, device=value.device)
        view = storage.as_strided(value.shape, (row + 256, *value.stride()[1:]))
        view.copy_(value)
        return view

    layer = SimpleNamespace(
        num_spec=0,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        num_k_heads=nk,
        num_v_heads=nv,
        tp_size=1,
        key_dim=nk * 128,
        value_dim=nv * 128,
        activation="silu",
        conv1d=SimpleNamespace(bias=None),
        norm=SimpleNamespace(
            eps=1e-6, norm_before_gate=True, group_size=None, activation="silu", weight=rand((128,), 0.3)
        ),
        A_log=rand((nv,), 0.2, torch.float32),
        _mega_gdn_dt_bias=rand((nv,), 0.2, torch.float32),
        kv_cache=(cache(conv_initial), cache(ssm_initial)),
    )
    layer.dt_bias = layer._mega_gdn_dt_bias
    indices = torch.arange(batch, 0, -1, dtype=torch.int32).npu()
    meta = SimpleNamespace(
        num_prefills=0,
        num_spec_decodes=0,
        spec_sequence_masks=None,
        num_decodes=batch,
        non_spec_state_indices_tensor=indices,
    )
    qkv, z, b, a = rand((batch, channels)), rand((batch, nv, 128)), rand((batch, nv)), rand((batch, nv))
    weight = rand((4, channels))
    output = torch.empty_like(z)
    return layer, meta, qkv, z, b, a, weight, output


def _golden(args, conv, state):
    layer, meta, qkv, z, b, a, weight, _ = args
    qkv, z, b, a, weight = [x.cpu().float() for x in (qkv, z, b, a, weight)]
    ids = meta.non_spec_state_indices_tensor.cpu().tolist()
    nv, nk = layer.num_v_heads, layer.num_k_heads
    outputs = []
    for row, slot in enumerate(ids):
        if slot == 0:
            outputs.append(torch.zeros(nv, 128, dtype=torch.bfloat16))
            continue
        history = conv[slot].float()
        c = F.silu((history * weight[:3]).sum(0) + qkv[row] * weight[3]).bfloat16().float()
        conv[slot, :2] = conv[slot, 1:].clone()
        conv[slot, 2] = qkv[row]
        q, k, v = c.split([nk * 128, nk * 128, nv * 128])
        q = q.reshape(nk, 128)
        k = k.reshape(nk, 128)
        v = v.reshape(nv, 128)
        q = (q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) / 128**0.5).repeat_interleave(nv // nk, 0)
        k = (k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)).repeat_interleave(nv // nk, 0)
        decay = (-layer.A_log.cpu().exp() * F.softplus(a[row] + layer.dt_bias.cpu(), threshold=20)).exp()
        beta = b[row].sigmoid()
        h = state[slot] * decay[:, None, None]
        delta = (v - (h * k[:, None, :]).sum(-1)) * beta[:, None]
        h = h + delta[:, :, None] * k[:, None, :]
        state[slot] = h
        o = (h * q[:, None, :]).sum(-1).bfloat16().float()
        y = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + 1e-6)
        y = y * layer.norm.weight.cpu().float() * F.silu(z[row])
        outputs.append(y.bfloat16())
    return torch.stack(outputs)


@pytest.mark.parametrize(
    "batch,nk,nv,padded", [(1, 1, 1, False), (4, 2, 4, False), (4, 2, 4, True), (16, 4, 12, True), (32, 1, 4, False)]
)
def test_output_and_state_against_reference(batch, nk, nv, padded):
    args = _case(batch, nk, nv, padded)
    layer = args[0]
    assert supports_mega_gdn_decode(*args)
    conv = layer.kv_cache[0].cpu().clone()
    state = layer.kv_cache[1].cpu().clone()
    expected = _golden(args, conv, state)
    run_mega_gdn_decode(*args)
    torch.npu.synchronize()
    torch.testing.assert_close(args[-1].cpu(), expected, rtol=0.03, atol=0.012)
    torch.testing.assert_close(layer.kv_cache[0].cpu(), conv, rtol=0, atol=0)
    torch.testing.assert_close(layer.kv_cache[1].cpu(), state, rtol=0.003, atol=0.0003)


@pytest.mark.parametrize("padded", [False, True])
def test_graph_replay_changes_inputs_and_indices(padded):
    args = list(_case(padded=padded))
    layer, meta = args[:2]
    for _ in range(2):
        run_mega_gdn_decode(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run_mega_gdn_decode(*args)
    torch.npu.synchronize()
    conv = layer.kv_cache[0].cpu().clone()
    state = layer.kv_cache[1].cpu().clone()
    for step in range(5):
        args[2].copy_((torch.randn(args[2].shape) * 0.2).bfloat16().npu())
        meta.non_spec_state_indices_tensor.copy_(torch.tensor([2, 4, 1, 3], dtype=torch.int32).npu())
        expected = _golden(args, conv, state)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(args[-1].cpu(), expected, rtol=0.04, atol=0.015)
        torch.testing.assert_close(layer.kv_cache[0].cpu(), conv, rtol=0, atol=0)
        torch.testing.assert_close(layer.kv_cache[1].cpu(), state, rtol=0.01, atol=0.0005)


def test_padding_preserves_live_cache_and_page_gaps():
    args = list(_case(padded=True))
    layer, meta = args[:2]
    meta.non_spec_state_indices_tensor.copy_(torch.tensor([4, 2, 0, 0], dtype=torch.int32).npu())
    conv = layer.kv_cache[0].cpu().clone()
    state = layer.kv_cache[1].cpu().clone()
    expected = _golden(args, conv, state)
    run_mega_gdn_decode(*args)
    torch.npu.synchronize()
    torch.testing.assert_close(args[-1][:2].cpu(), expected[:2], rtol=0.03, atol=0.012)
    torch.testing.assert_close(layer.kv_cache[0].cpu(), conv, rtol=0, atol=0)
    torch.testing.assert_close(layer.kv_cache[1].cpu(), state, rtol=0.003, atol=0.0003)
    for cache in layer.kv_cache:
        row = cache[0].numel()
        stride = cache.stride(0)
        gaps = cache.as_strided((cache.shape[0], stride - row), (stride, 1), storage_offset=row)
        assert torch.all(gaps.cpu() == 9.0)


def test_against_existing_decode_core(monkeypatch):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    import vllm_ascend.ops.gdn as gdn
    from vllm_ascend.ops.gdn_attn_builder import GDNCausalConv1dMetadata, GDNDecodeMetadata
    from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch

    args = list(_case(padded=True))
    layer, meta = args[:2]
    baseline = copy.copy(layer)
    baseline.kv_cache = tuple(t.clone() for t in layer.kv_cache)
    baseline.prefix = "test.linear_attn"
    query_start = torch.arange(5, dtype=torch.int32).npu()
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=4,
        num_decode_tokens=4,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=4,
        spec_sequence_masks=None,
        non_spec_state_indices_tensor=meta.non_spec_state_indices_tensor,
        non_spec_query_start_loc=query_start,
    )
    metadata.non_spec_decode_metadata = GDNDecodeMetadata(
        causal_conv1d=GDNCausalConv1dMetadata(
            query_start_loc=query_start, cache_indices=meta.non_spec_state_indices_tensor, initial_state_mode=None
        ),
        actual_seq_lengths=torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32).npu(),
    )
    monkeypatch.setattr(gdn, "get_forward_context", lambda: SimpleNamespace(attn_metadata={baseline.prefix: metadata}))
    monkeypatch.setattr(gdn, "get_weight_prefetch_method", lambda: None)
    monkeypatch.setattr(gdn, "wait_for_kv_layer_from_connector", lambda *args: None)
    monkeypatch.setattr(gdn, "record_attention_compute_start", lambda: None)
    monkeypatch.setattr(gdn, "maybe_save_kv_layer_to_connector", lambda *args: None)
    monkeypatch.setattr(gdn, "_get_packed_conv_weights", lambda layer: args[6])
    monkeypatch.setattr(gdn, "_gate_norm_fusion_enabled", lambda: False)
    monkeypatch.setattr(gdn.DeviceOperator, "fused_gdn_gating", fused_gdn_gating_patch)
    original_out = torch.zeros_like(args[-1])
    for step in range(12):
        args[2].copy_((torch.randn(args[2].shape) * 0.2).bfloat16().npu())
        gdn.AscendGatedDeltaNetAttention._forward_core(baseline, args[2], args[4], args[5], original_out)
        o = original_out.float()
        expected = (
            o
            * torch.rsqrt(o.square().mean(-1, keepdim=True) + 1e-6)
            * layer.norm.weight.float()
            * F.silu(args[3].float())
        ).bfloat16()
        run_mega_gdn_decode(*args)
        torch.npu.synchronize()
        delta = (args[-1].float() - expected.float()).abs()
        cosine = F.cosine_similarity(args[-1].float().flatten(), expected.float().flatten(), dim=0)
        print("BASELINE", step, "max_abs", delta.max().item(), "cosine", cosine.item())
        torch.testing.assert_close(args[-1].cpu(), expected.cpu(), rtol=0.05, atol=0.025)
        torch.testing.assert_close(layer.kv_cache[0], baseline.kv_cache[0], rtol=0, atol=0)
        torch.testing.assert_close(layer.kv_cache[1], baseline.kv_cache[1], rtol=0.05, atol=0.004)
        assert cosine.item() > 0.999
