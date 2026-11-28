# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.ops.triton.fla.split_gdn_projection import split_gdn_projection


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_single_token_projection_split_is_a_view(device):
    if device == "npu" and not torch.npu.is_available():
        pytest.skip("NPU is not available")
    qkvz = torch.arange(4096, device=device, dtype=torch.float32).reshape(1, 4096)
    ba = torch.arange(24, device=device, dtype=torch.float32).reshape(1, 24)

    qkv, z, b, a = split_gdn_projection(qkvz, ba, 2560, 1536, 128)

    expected = (
        qkvz[:, :2560],
        qkvz[:, 2560:].reshape(1, 12, 128),
        ba[:, :12],
        ba[:, 12:],
    )
    for actual, reference in zip((qkv, z, b, a), expected):
        torch.testing.assert_close(actual, reference)
        assert actual.is_contiguous()

    assert qkv.data_ptr() == qkvz.data_ptr()
    assert z.data_ptr() == qkvz.data_ptr() + 2560 * qkvz.element_size()
    assert b.data_ptr() == ba.data_ptr()
    assert a.data_ptr() == ba.data_ptr() + 12 * ba.element_size()
