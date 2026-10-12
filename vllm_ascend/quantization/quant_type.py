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
"""Shared quantization enum definitions.

Keep this module lightweight and side-effect free so core runtime modules can
import QuantType without triggering heavy quantization package initialization.
"""

from enum import Enum


class QuantType(Enum):
    """Quantization type enum for MoE schemes."""

    NONE = 0
    W8A8 = 1
    W4A8 = 2
    MXFP8 = 3
    W4A16 = 4
    MXFP4 = 5
    W4A8MXFP = 6
    W8A8FP8 = 7
    W4A16MXFP4 = 8



# Quant types whose weight layouts the A5 MegaMoe (FUSED_MC2) operator supports.
# NOTE: the names here differ from the vllm-ascend main branch (which calls these
# W8A8MXFP / W4A4MXFP); this tree uses MXFP8 / MXFP4 for the same scenarios:
#   MXFP8     -> A8W8-FP (act FP8, weight FP8)
#   W4A8MXFP  -> A8W4-FP (act FP8, weight FP4)
#   MXFP4     -> A4W4-FP (act FP4, weight FP4)
A5_SUPPORT_MEGA_MOE_QUANT_TYPES = frozenset(
    {
        QuantType.MXFP8,
        QuantType.W4A8MXFP,
        QuantType.MXFP4,
    }
)
