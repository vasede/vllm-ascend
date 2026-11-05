# SPDX-License-Identifier: Apache-2.0
"""Compare the existing GDN core + norm with MegaGdnDecode on one NPU."""

import argparse
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--key-heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=None,
        help="Optional graph-latency regression limit for a controlled device",
    )
    options = parser.parse_args()
    test_path = (
        Path(__file__).resolve().parents[1] / "tests/e2e/nightly/single_node/ops/singlecard_ops/test_mega_gdn_decode.py"
    )
    spec = importlib.util.spec_from_file_location("mega_gdn_cases", test_path)
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)

    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    import vllm_ascend.ops.gdn as gdn
    from vllm_ascend.ops.gdn_attn_builder import GDNCausalConv1dMetadata, GDNDecodeMetadata
    from vllm_ascend.ops.layernorm import LayerNormFn
    from vllm_ascend.ops.mega_gdn_decode import run_mega_gdn_decode
    from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    def elapsed(fn):
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(options.repeats):
            fn()
        end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end) * 1000 / options.repeats

    def graph(fn):
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        result = torch.npu.NPUGraph()
        with torch.npu.graph(result):
            fn()
        return result

    print("cache,existing_eager_us,mega_eager_us,existing_graph_us,mega_graph_us,graph_speedup")
    for padded in (False, True):
        args = cases._case(options.batch, options.key_heads, options.value_heads, padded)
        init_device_properties_triton()
        layer, metadata, qkv, z, b, a, weight, _ = args
        baseline = copy.copy(layer)

        def clone_state(value):
            storage = torch.empty((value.shape[0], value.stride(0)), dtype=value.dtype, device=value.device)
            result = storage.as_strided(value.shape, value.stride())
            result.copy_(value)
            return result

        baseline.kv_cache = tuple(clone_state(t) for t in layer.kv_cache)
        baseline.prefix = "benchmark.linear_attn"
        batch = options.batch
        indices = metadata.non_spec_state_indices_tensor
        query_start = torch.arange(batch + 1, dtype=torch.int32).npu()
        lengths = torch.cat((torch.zeros(1, dtype=torch.int32), torch.ones(batch, dtype=torch.int32))).npu()
        meta = GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=batch,
            num_decode_tokens=batch,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=batch,
            spec_sequence_masks=None,
            non_spec_state_indices_tensor=indices,
            non_spec_query_start_loc=query_start,
        )
        meta.non_spec_decode_metadata = GDNDecodeMetadata(
            causal_conv1d=GDNCausalConv1dMetadata(
                query_start_loc=query_start, cache_indices=indices, initial_state_mode=None
            ),
            actual_seq_lengths=lengths,
        )
        raw = torch.zeros_like(z)

        def old(baseline=baseline, qkv=qkv, b=b, a=a, raw=raw, layer=layer, z=z):
            gdn.AscendGatedDeltaNetAttention._forward_core(baseline, qkv, b, a, raw)
            return LayerNormFn.apply(
                raw.reshape(-1, 128), layer.norm.weight, None, z.reshape(-1, 128), 1e-6, None, True, True
            )

        def new(args=args):
            run_mega_gdn_decode(*args)

        with (
            patch.object(
                gdn, "get_forward_context", return_value=SimpleNamespace(attn_metadata={baseline.prefix: meta})
            ),
            patch.object(gdn, "get_weight_prefetch_method", return_value=None),
            patch.object(gdn, "wait_for_kv_layer_from_connector"),
            patch.object(gdn, "record_attention_compute_start"),
            patch.object(gdn, "maybe_save_kv_layer_to_connector"),
            patch.object(gdn, "_get_packed_conv_weights", return_value=weight),
            patch.object(gdn, "_gate_norm_fusion_enabled", return_value=False),
            patch.object(gdn.DeviceOperator, "fused_gdn_gating", fused_gdn_gating_patch),
        ):
            old_eager, new_eager = elapsed(old), elapsed(new)
            old_graph, new_graph = graph(old), graph(new)
            old_us, new_us = elapsed(old_graph.replay), elapsed(new_graph.replay)
        print(
            f"{'strided' if padded else 'contiguous'},{old_eager:.3f},{new_eager:.3f},"
            f"{old_us:.3f},{new_us:.3f},{old_us / new_us:.3f}",
            flush=True,
        )
        if options.max_slowdown is not None:
            assert new_us / old_us <= options.max_slowdown


if __name__ == "__main__":
    main()
