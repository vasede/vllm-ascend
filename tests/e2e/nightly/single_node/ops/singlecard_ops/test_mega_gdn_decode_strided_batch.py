# SPDX-License-Identifier: Apache-2.0
"""B1..32 direct state update and captured graph regression on one NPU."""

import gc
import json

import pytest
import torch


@pytest.fixture(autouse=True)
def require_batch_op():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    if (
        not hasattr(torch.ops._C_ascend, "has_mega_gdn_decode_strided_batch")
        or not torch.ops._C_ascend.has_mega_gdn_decode_strided_batch()
    ):
        pytest.skip("Source the MegaGdnDecodeStridedBatch OPP environment")
    torch.npu.set_device(0)
    torch.manual_seed(43)
    yield
    gc.collect()
    torch.npu.empty_cache()


def run(batch, nk=4, nv=12, slots=96, stride_extra=256, graph_mode=False, wide=False, layout=False):
    c = (2 * nk + nv) * 128
    cr = 3 * c
    sr = nv * 128 * 128
    cs = cr + stride_extra
    ss = 524288 if wide else sr + stride_extra
    stor_c = torch.full(((slots - 1) * cs + cr + 64,), 9, dtype=torch.bfloat16, device="npu")
    stor_s = torch.empty((slots - 1) * ss + sr + 64, dtype=torch.float32, device="npu")
    if slots < 1024:
        stor_s.fill_(7)
    conv = stor_c.as_strided((slots, 3, c), (cs, c, 1), storage_offset=32)
    state = stor_s.as_strided((slots, nv, 128, 128), (ss, 128 * 128, 128, 1), storage_offset=32)

    def rand(shape, scale=0.2, dtype=torch.bfloat16):
        return (torch.randn(shape) * scale).to(dtype).npu()

    w = rand((4, c))
    alog = rand((nv,), 0.2, torch.float32)
    dt = rand((nv,), 0.2, torch.float32)
    norm = rand((128,), 0.3)
    qkv = rand((batch, c))
    z = rand((batch, nv, 128))
    a = rand((batch, nv))
    b = rand((batch, nv))
    ids = torch.empty(batch, dtype=torch.int32, device="npu")
    write_ids = torch.empty_like(ids)
    cy = torch.empty_like(qkv)
    y = torch.empty_like(z)
    ref_c = torch.empty((batch + 1, 3, c), dtype=torch.bfloat16, device="npu")
    ref_s = torch.empty((batch + 1, nv, 128, 128), dtype=torch.float32, device="npu")
    ref_ids = torch.arange(1, batch + 1, dtype=torch.int32, device="npu")
    ref_cy = torch.empty_like(cy)
    ref_y = torch.empty_like(y)
    initial_c = rand((batch, 3, c))
    initial_s = rand((batch, nv, 128, 128), 0.05, torch.float32)

    def native():
        torch.ops._C_ascend.mega_gdn_decode_strided_batch(
            qkv, z, b, a, w, conv, alog, dt, state, ids, write_ids, norm, cy, y, layout
        )

    def reference():
        torch.ops._C_ascend.mega_gdn_decode(
            qkv, z, b, a, w, ref_c, alog, dt, ref_s, ref_ids, ref_ids, norm, ref_cy, ref_y, layout
        )

    # Distinct high slots, in reverse request order, never assuming physical adjacency.
    live_ids = list(range(slots - batch, slots))[::-1]
    for j, slot in enumerate(live_ids):
        conv[slot].copy_(initial_c[j])
        state[slot].copy_(initial_s[j])
    ref_c[1:].copy_(initial_c)
    ref_s[1:].copy_(initial_s)
    ids.copy_(torch.tensor(live_ids, dtype=torch.int32).npu())
    write_ids.copy_(ids)
    # Warmup and graph capture can mutate state; reset each pattern below.
    graph = None
    if graph_mode:
        native()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            native()
        torch.npu.synchronize()
    patterns = [list(range(batch)), list(range(batch // 2, batch)), list(range(0, batch, 2)), [], list(range(batch))]
    for pattern_idx, active_rows in enumerate(patterns):
        act = set(active_rows)
        # Interspersed padding, not only a tail. Shared slot 0 must never be touched.
        current = [live_ids[j] if j in act else (0, -1, slots, slots + 1)[j % 4] for j in range(batch)]
        for j, slot in enumerate(live_ids):
            conv[slot].copy_(initial_c[j])
            state[slot].copy_(initial_s[j])
        conv[0].fill_(5)
        state[0].fill_(6)
        ref_c[1:].copy_(initial_c)
        ref_s[1:].copy_(initial_s)
        ids.copy_(torch.tensor(current, dtype=torch.int32).npu())
        write_ids.copy_(ids)
        for step in range(3):
            qkv.copy_(rand(qkv.shape))
            reference()
            if graph is None:
                native()
            else:
                graph.replay()
            torch.npu.synchronize()
            for j, slot in enumerate(live_ids):
                if j in act:
                    torch.testing.assert_close(y[j], ref_y[j], rtol=0, atol=0)
                    torch.testing.assert_close(cy[j], ref_cy[j], rtol=0, atol=0)
                    torch.testing.assert_close(conv[slot], ref_c[j + 1], rtol=0, atol=0)
                    torch.testing.assert_close(state[slot], ref_s[j + 1], rtol=0, atol=0)
                else:
                    assert torch.count_nonzero(y[j].cpu()) == 0 and torch.count_nonzero(cy[j].cpu()) == 0
                    torch.testing.assert_close(conv[slot], initial_c[j], rtol=0, atol=0)
                    torch.testing.assert_close(state[slot], initial_s[j], rtol=0, atol=0)
            assert torch.all(conv[0].cpu() == 5) and torch.all(state[0].cpu() == 6)
        # Move request order without recapturing, while retaining each request's slot.
        live_ids = live_ids[1:] + live_ids[:1]
    # Prefix fork: many requests can read one readonly source, all writes private.
    shared = 1
    conv[shared].copy_(initial_c[0])
    state[shared].copy_(initial_s[0])
    ref_c[1:].copy_(initial_c[0].expand_as(ref_c[1:]))
    ref_s[1:].copy_(initial_s[0].expand_as(ref_s[1:]))
    ids.fill_(shared)
    write_ids.copy_(torch.tensor(live_ids, dtype=torch.int32).npu())
    reference()
    if graph is None:
        native()
    else:
        graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
    for j, slot in enumerate(live_ids):
        torch.testing.assert_close(conv[slot], ref_c[j + 1], rtol=0, atol=0)
        torch.testing.assert_close(state[slot], ref_s[j + 1], rtol=0, atol=0)
    torch.testing.assert_close(conv[shared], initial_c[0], rtol=0, atol=0)
    torch.testing.assert_close(state[shared], initial_s[0], rtol=0, atol=0)
    # Padded page gaps and view prefix remain untouched.
    for slot in live_ids if stride_extra else []:
        assert torch.all(stor_c[32 + slot * cs - 16 : 32 + slot * cs].cpu() == 9)
        if slots < 1024:
            assert torch.all(stor_s[32 + slot * ss - 16 : 32 + slot * ss].cpu() == 7)
    print(
        "PASS",
        json.dumps(
            dict(
                batch=batch,
                nk=nk,
                nv=nv,
                slots=slots,
                graph=graph_mode,
                wide=wide,
                layout=layout,
                patterns=len(patterns),
            )
        ),
        flush=True,
    )


@pytest.mark.parametrize("batch", range(1, 33))
def test_all_batch_sizes_graph_padding_and_request_reorder(batch):
    run(batch, graph_mode=True)


@pytest.mark.parametrize("batch,nk,nv", [(2, 1, 1), (3, 2, 4), (8, 8, 24), (16, 16, 48), (32, 16, 64)])
def test_multiple_head_geometries(batch, nk, nv):
    run(batch, nk, nv, graph_mode=True)


@pytest.mark.parametrize("batch", [2, 8, 16, 32])
def test_large_state_pool(batch):
    run(batch, slots=5800, graph_mode=True)


def test_offset_above_int32():
    run(3, slots=5800, wide=True, graph_mode=True)


@pytest.mark.parametrize("batch", [2, 8, 32])
def test_fla_state_layout(batch):
    run(batch, layout=True, graph_mode=True)


def test_contiguous_states_eager():
    run(7, stride_extra=0)
