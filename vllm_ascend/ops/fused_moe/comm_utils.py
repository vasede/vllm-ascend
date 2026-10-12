# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
from importlib import import_module

import torch
import torch.distributed
import torch.distributed as dist
import torch_npu
from vllm.logger import logger

from vllm_ascend.quantization.quant_type import QuantType

COMM_STREAM = None


def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None):
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.npu.current_device(),
        )

    if event:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
    else:
        handle = dist.all_to_all_single(
            a2a_out,
            input_.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
    return input_, a2a_out, handle


def _gather_along_first_dim(input_, group, output_split_sizes=None):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.npu.current_device())
        torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.npu.current_device())
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def gather_from_sequence_parallel_region(
    input_,
    group,
    output_split_sizes=None,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    return _gather_along_first_dim(input_, group, output_split_sizes)




# ---------------------------------------------------------------------------
# CANN MegaMoe (mc2/mega_moe) integration helpers.
#
# MegaMoe fuses dispatch + grouped-matmul + swiglu + combine into one op. It is
# delivered by the CANN ops-transformer component, whose python entry points
# live in the ``cann_ops_transformer.ops`` module (NOT torch_npu). The constants
# below are the ACL dtype / quant-mode enums the op expects; the authoritative
# reference is the CANN-shipped
# $ASCEND_TOOLKIT_HOME/python/site-packages/cann_ops_transformer/docs/zh/mega_moe.md
# ---------------------------------------------------------------------------
_CANN_ACL_INT8 = 258
_CANN_ACL_INT4 = 285
# 23 = FLOAT8_E5M2, 24 = FLOAT8_E4M3FN, 296 = FLOAT4_E2M1.
_CANN_ACL_FLOAT8_E5M2 = 23
_CANN_ACL_FLOAT8_E4M3FN = 24
_CANN_ACL_FLOAT4_E2M1 = 296

_CANN_MEGA_MOE_QUANT_MODE_NONE = 0
_CANN_MEGA_MOE_QUANT_MODE_INT8 = 2
_CANN_MEGA_MOE_QUANT_MODE_MXFP = 4

# QuantType -> (dispatch_quant_mode, dispatch_quant_out_dtype, weight_type).
#
# A5 (Ascend 950PR/DT) is MXFP-only: the doc states A16W16 / A8W8-INT /
# A8W4-INT are NOT supported, and dispatch_quant_mode must be 4. The
# dispatch_quant_out_dtype binds to the ACTIVATION dtype and weight_type to the
# WEIGHT dtype, per the A5 scenario table:
#   MXFP8    -> A8W8-FP (act FP8, weight FP8)
#   W4A8MXFP -> A8W4-FP (act FP8, weight FP4)
#   MXFP4    -> A4W4-FP (act FP4, weight FP4)
# NOTE: this tree's QuantType names differ from vllm-ascend main (MXFP8 here ==
# W8A8MXFP upstream, MXFP4 here == W4A4MXFP upstream).
#
# The INT8/INT4 rows are the A2/A3 scenarios, kept for completeness; the ACL
# enum ints (258 / 285) are passed rather than torch dtypes because the op
# derives W4A8's packed N from the int8 storage shape.
_CANN_MEGA_MOE_QUANT_SETTINGS: dict[QuantType, tuple[int, int | None, int | None]] = {
    QuantType.NONE: (_CANN_MEGA_MOE_QUANT_MODE_NONE, None, None),
    QuantType.W8A8: (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT8),
    QuantType.W4A8: (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT4),
    QuantType.MXFP8: (_CANN_MEGA_MOE_QUANT_MODE_MXFP, _CANN_ACL_FLOAT8_E4M3FN, _CANN_ACL_FLOAT8_E4M3FN),
    QuantType.W4A8MXFP: (_CANN_MEGA_MOE_QUANT_MODE_MXFP, _CANN_ACL_FLOAT8_E4M3FN, _CANN_ACL_FLOAT4_E2M1),
    QuantType.MXFP4: (_CANN_MEGA_MOE_QUANT_MODE_MXFP, _CANN_ACL_FLOAT4_E2M1, _CANN_ACL_FLOAT4_E2M1),
}


def load_cann_mega_moe_ops():
    """Return (get_symm_buffer_for_mega_moe, mega_moe) from the CANN package."""
    ops_module = import_module("cann_ops_transformer.ops")
    return ops_module.get_symm_buffer_for_mega_moe, ops_module.mega_moe


def get_cann_mega_moe_quant_settings(quant_type: QuantType) -> tuple[int, int | None, int | None]:
    """Returns (dispatch_quant_mode, dispatch_quant_out_dtype, weight_type)."""
    if quant_type not in _CANN_MEGA_MOE_QUANT_SETTINGS:
        raise RuntimeError(
            "MegaMoe integration supports W8A8/W4A8 INT on A2/A3 and MXFP8/W4A8MXFP/MXFP4 on A5. "
            f"Unsupported quant type: {quant_type}."
        )
    settings = _CANN_MEGA_MOE_QUANT_SETTINGS[quant_type]
    logger.debug(
        "cann mega_moe quant settings: dispatch_quant_mode=%s dispatch_quant_out_dtype=%s weight_type=%s",
        *settings,
    )
    return settings

