# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.ops.mega_gdn_decode as mega


def make_case(batch=4):
    nk, nv, channels, slots = 2, 4, 1024, batch + 3
    layer = SimpleNamespace(
        num_spec=0,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        num_k_heads=nk,
        num_v_heads=nv,
        tp_size=1,
        activation="silu",
        conv1d=SimpleNamespace(bias=None),
        norm=SimpleNamespace(
            eps=1e-6,
            norm_before_gate=True,
            group_size=None,
            activation="silu",
            weight=torch.ones(128, dtype=torch.bfloat16),
        ),
        A_log=torch.zeros(nv),
        _mega_gdn_dt_bias=torch.zeros(nv),
        kv_cache=(torch.zeros(slots, 3, channels, dtype=torch.bfloat16), torch.zeros(slots, nv, 128, 128)),
    )
    meta = SimpleNamespace(
        num_prefills=0,
        num_spec_decodes=0,
        spec_sequence_masks=None,
        num_decodes=batch,
        non_spec_state_indices_tensor=torch.arange(1, batch + 1, dtype=torch.int32),
    )
    qkv = torch.zeros(batch, channels, dtype=torch.bfloat16)
    z = torch.zeros(batch, nv, 128, dtype=torch.bfloat16)
    b = torch.zeros(batch, nv, dtype=torch.bfloat16)
    weight = torch.zeros(4, channels, dtype=torch.bfloat16)
    return layer, meta, qkv, z, b, b.clone(), weight, torch.zeros_like(z)


@pytest.mark.parametrize("batch", range(1, 33))
def test_native_batch_routes_padded_states_without_copy(monkeypatch, batch):
    args = make_case(batch)
    layer, meta = args[:2]
    # Graph padding keeps its device index and aliases the original state pool.
    meta.non_spec_state_indices_tensor[-1] = 0
    layer._mega_gdn_strided_batch_ready = True
    layer._mega_gdn_strided_ready = True
    layer.kv_cache = tuple(
        torch.empty_strided(t.shape, (t[0].numel() + 256, *t.stride()[1:]), dtype=t.dtype) for t in layer.kv_cache
    )
    native, old, single = Mock(), Mock(), Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "mega_gdn_decode_strided_batch", native, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "mega_gdn_decode", old, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "mega_gdn_decode_strided", single, raising=False)
    assert mega.supports_mega_gdn_decode(*args)
    mega.run_mega_gdn_decode(*args)
    native.assert_called_once()
    actual = native.call_args.args
    assert actual[5] is layer.kv_cache[0] and actual[8] is layer.kv_cache[1]
    assert actual[9].data_ptr() == meta.non_spec_state_indices_tensor.data_ptr()
    assert actual[13].data_ptr() == args[-1].data_ptr()
    old.assert_not_called()
    single.assert_not_called()


@pytest.mark.parametrize("batch,old_single", [(1, True), (1, False), (4, True), (32, True)])
def test_missing_batch_binary_retains_compatible_fallback(monkeypatch, batch, old_single):
    args = make_case(batch)
    args[0]._mega_gdn_strided_batch_ready = False
    args[0]._mega_gdn_strided_ready = old_single
    native, old, single = Mock(), Mock(), Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "mega_gdn_decode_strided_batch", native, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "mega_gdn_decode", old, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "mega_gdn_decode_strided", single, raising=False)
    mega.run_mega_gdn_decode(*args)
    native.assert_not_called()
    expected = single if batch == 1 and old_single else old
    expected.assert_called_once()
