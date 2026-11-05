# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch


@pytest.fixture(autouse=True)
def require_strided_op():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    if (
        not hasattr(torch.ops._C_ascend, "has_mega_gdn_decode_strided")
        or not torch.ops._C_ascend.has_mega_gdn_decode_strided()
    ):
        pytest.skip("Source the MegaGdnDecodeStrided OPP environment")
    torch.npu.set_device(0)
    torch.manual_seed(431)


def invoke(op, args, conv, ssm, ids, out, convout):
    qkv, z, b, a, w, alog, dt, norm = args
    op(qkv, z, b, a, w, conv, alog, dt, ssm, ids, ids, norm, convout, out, False)


def run(slots, nk, nv, ssm_stride=None, conv_stride=None, slot=None, steps=8):
    c = (2 * nk + nv) * 128
    sr = nv * 128 * 128
    cr = 3 * c
    ss = ssm_stride or sr + 1024
    cs = conv_stride or cr + 128
    slot = slot or slots - 1
    # Nonzero storage offsets test subviews of the physical backing allocations.
    sraw = torch.empty((slots - 1) * ss + sr + 96, dtype=torch.float32, device="npu")
    craw = torch.empty((slots - 1) * cs + cr + 96, dtype=torch.bfloat16, device="npu")
    ssm = sraw.as_strided((slots, nv, 128, 128), (ss, 128 * 128, 128, 1), storage_offset=32)
    conv = craw.as_strided((slots, 3, c), (cs, c, 1), storage_offset=32)

    def rand(shape, scale=0.2, dtype=torch.bfloat16):
        return (torch.randn(shape) * scale).to(dtype).npu()

    initial_c = rand((3, c))
    initial_s = rand((nv, 128, 128), 0.05, torch.float32)
    conv[slot].copy_(initial_c)
    ssm[slot].copy_(initial_s)
    old_c = torch.stack([torch.zeros_like(initial_c), initial_c])
    old_s = torch.stack([torch.zeros_like(initial_s), initial_s])
    args = [
        rand((1, c)),
        rand((1, nv, 128)),
        rand((1, nv)),
        rand((1, nv)),
        rand((4, c)),
        rand((nv,), 0.2, torch.float32),
        rand((nv,), 0.2, torch.float32),
        rand((128,), 0.3),
    ]
    ids = torch.tensor([slot], dtype=torch.int32, device="npu")
    one = torch.ones(1, dtype=torch.int32, device="npu")
    y = torch.empty_like(args[1])
    yc = torch.empty_like(args[0])
    old_y = torch.empty_like(y)
    old_yc = torch.empty_like(yc)
    sentinel_s = sraw[32 + slot * ss - 16 : 32 + slot * ss]
    sentinel_c = craw[32 + slot * cs - 16 : 32 + slot * cs]
    sentinel_s.fill_(7)
    sentinel_c.fill_(9)
    sraw[:32].fill_(11)
    craw[:32].fill_(13)
    for step in range(steps):
        args[0].copy_(rand(args[0].shape))
        invoke(torch.ops._C_ascend.mega_gdn_decode, args, old_c, old_s, one, old_y, old_yc)
        invoke(torch.ops._C_ascend.mega_gdn_decode_strided, args, conv, ssm, ids, y, yc)
        torch.npu.synchronize()
        torch.testing.assert_close(y, old_y, rtol=0, atol=0)
        torch.testing.assert_close(yc, old_yc, rtol=0, atol=0)
        torch.testing.assert_close(conv[slot], old_c[1], rtol=0, atol=0)
        torch.testing.assert_close(ssm[slot], old_s[1], rtol=0, atol=0)
    assert torch.all(sentinel_s.cpu() == 7) and torch.all(sentinel_c.cpu() == 9)
    assert torch.all(sraw[:32].cpu() == 11) and torch.all(craw[:32].cpu() == 13)
    # Graph replay must read device index values at replay time, including padding.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        invoke(torch.ops._C_ascend.mega_gdn_decode_strided, args, conv, ssm, ids, y, yc)
    torch.npu.synchronize()
    for index in [1, slot, 0, slot]:
        if index > 0:
            conv[index].copy_(initial_c)
            ssm[index].copy_(initial_s)
            old_c[1].copy_(initial_c)
            old_s[1].copy_(initial_s)
            invoke(torch.ops._C_ascend.mega_gdn_decode, args, old_c, old_s, one, old_y, old_yc)
        ids.fill_(index)
        graph.replay()
        torch.npu.synchronize()
        if index == 0:
            print(
                "PAD_GRAPH",
                ids.cpu().tolist(),
                "y_nonzero",
                torch.count_nonzero(y.cpu()).item(),
                "c_nonzero",
                torch.count_nonzero(yc.cpu()).item(),
                flush=True,
            )
            assert torch.count_nonzero(y.cpu()) == 0 and torch.count_nonzero(yc.cpu()) == 0
            invoke(torch.ops._C_ascend.mega_gdn_decode_strided, args, conv, ssm, ids, y, yc)
            torch.npu.synchronize()
            print(
                "PAD_EAGER",
                "y_nonzero",
                torch.count_nonzero(y.cpu()).item(),
                "c_nonzero",
                torch.count_nonzero(yc.cpu()).item(),
                flush=True,
            )
            assert torch.count_nonzero(y.cpu()) == 0 and torch.count_nonzero(yc.cpu()) == 0
        else:
            torch.testing.assert_close(y, old_y, rtol=0, atol=0)
            torch.testing.assert_close(ssm[index], old_s[1], rtol=0, atol=0)
    print(
        "PASS",
        json.dumps(
            dict(
                slots=slots,
                nk=nk,
                nv=nv,
                slot=slot,
                ssm_stride=ss,
                conv_stride=cs,
                max_state_offset_elements=slot * ss,
                steps=steps,
                graph=True,
            )
        ),
        flush=True,
    )


@pytest.mark.parametrize("slots,nk,nv", [(8, 1, 1), (8, 4, 12), (8, 8, 24), (8, 16, 48), (8, 16, 64), (5800, 4, 12)])
def test_direct_strided_state_matches_dense_and_graph(slots, nk, nv):
    run(slots, nk, nv)


def test_state_address_above_int32_elements():
    run(5800, 4, 12, ssm_stride=524288, steps=3)
