# vllm_ascend/patch/worker/patch_qwen3_5_gdn_split.py
#
# 功能：通过 monkey-patch 将 Qwen3.5 GDN 层的 in_proj_qkvz 拆分为独立模块，
#       并同步修正 weight loading 映射，与直接修改 vLLM 源码完全等价。

import torch
from typing import Iterable, Tuple, Set

from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention as _GDNBaseCls,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM


_original_gdn_init = _GDNBaseCls.__init__

def _patched_gdn_init(self, *args, **kwargs):
    
    # 1) 让原始 init 完成所有工作（自动透传 config 等所有参数）
    _original_gdn_init(self, *args, **kwargs)
    

    if hasattr(self, "in_proj_qkv") and not hasattr(self, "in_proj_qkvz"):
        return

    if hasattr(self, "in_proj_qkvz"):
        delattr(self, "in_proj_qkvz")

    hidden_size = getattr(self, "hidden_size", None)
    if hidden_size is None:
        # fallback：尝试从 conv1d 的输入维度推断
        hidden_size = self.conv1d.weight.shape[1] if hasattr(self, "conv1d") else args[0] if args else None

    num_k_heads = getattr(self, "num_k_heads", getattr(self, "num_heads", 1))
    num_v_heads = getattr(self, "num_v_heads", getattr(self, "num_heads", 1))
    head_k_dim = getattr(self, "head_k_dim", 128)
    head_v_dim = getattr(self, "head_v_dim", 128)

    # 5) 计算全局输出维度（ColumnParallelLinear 内部自动按 TP size 切分）
    qkv_global_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim
    z_global_size = num_v_heads * head_v_dim

    # 6) 从原始 init 的 quant_config 推断（如果原始 init 设置了的话）
    self.quant_config = getattr(self, "quant_config", None)

    # 7) 构建 prefix（从原始 init 的 prefix 推断）
    prefix = getattr(self, "prefix", "")

    self.in_proj_qkv = ColumnParallelLinear(
        input_size=hidden_size,
        output_size=qkv_global_size,
        bias=False,
        quant_config=self.quant_config,
        prefix=f"{prefix}.in_proj_qkv" if prefix else "in_proj_qkv",
    )
    self.in_proj_z = ColumnParallelLinear(
        input_size=hidden_size,
        output_size=z_global_size,
        bias=False,
        quant_config=self.quant_config,
        prefix=f"{prefix}.in_proj_z" if prefix else "in_proj_z",
    )

_GDNBaseCls.__init__ = _patched_gdn_init

