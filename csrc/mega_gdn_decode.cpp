// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

#include <torch/library.h>
#include "aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {
void mega_gdn_decode(
    const at::Tensor& qkv, const at::Tensor& z, const at::Tensor& b,
    const at::Tensor& a, const at::Tensor& conv_weight,
    const at::Tensor& conv_state, const at::Tensor& a_log,
    const at::Tensor& dt_bias, const at::Tensor& ssm_state,
    const at::Tensor& read_indices, const at::Tensor& write_indices,
    const at::Tensor& norm_weight, const at::Tensor& conv_out,
    const at::Tensor& out, bool fla_ssm_state_layout) {
  for (const auto& tensor : {qkv, z, b, a, conv_weight, conv_state,
                             a_log, dt_bias, ssm_state, read_indices,
                             write_indices, norm_weight, conv_out, out}) {
    TORCH_CHECK(tensor.is_contiguous(), "MegaGdnDecode requires contiguous tensors");
    TORCH_CHECK(tensor.device() == qkv.device(), "MegaGdnDecode device mismatch");
  }
  for (const auto& tensor : {qkv, z, b, a, conv_weight, conv_state,
                             norm_weight, conv_out, out}) {
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, "MegaGdnDecode requires BF16 activations");
  }
  for (const auto& tensor : {a_log, dt_bias, ssm_state}) {
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, "MegaGdnDecode requires FP32 state/decay weights");
  }
  TORCH_CHECK(read_indices.scalar_type() == at::kInt && write_indices.scalar_type() == at::kInt,
              "MegaGdnDecode indices must be INT32");
  TORCH_CHECK(conv_out.sizes() == qkv.sizes() && out.sizes() == z.sizes(), "MegaGdnDecode output shape mismatch");
  EXEC_NPU_CMD(aclnnMegaGdnDecode, qkv, z, b, a, conv_weight, conv_state,
               a_log, dt_bias, ssm_state, read_indices, write_indices,
               norm_weight, fla_ssm_state_layout, conv_out, conv_state, ssm_state, out);
}
}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
  m.def("mega_gdn_decode(Tensor qkv, Tensor z, Tensor b, Tensor a, Tensor conv_weight, "
        "Tensor(a!) conv_state, Tensor a_log, Tensor dt_bias, Tensor(b!) ssm_state, "
        "Tensor read_indices, Tensor write_indices, Tensor norm_weight, "
        "Tensor(c!) conv_out, Tensor(d!) out, bool fla_ssm_state_layout=True) -> ()");
  m.impl("mega_gdn_decode", torch::kPrivateUse1, &vllm_ascend::mega_gdn_decode);
}
