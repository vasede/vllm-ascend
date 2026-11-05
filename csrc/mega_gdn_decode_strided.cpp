// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#include <limits>
#include <torch/library.h>
#include "aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {
bool has_mega_gdn_decode_strided() {
  return GetOpApiFuncAddr("aclnnMegaGdnDecodeStrided") != nullptr &&
         GetOpApiFuncAddr("aclnnMegaGdnDecodeStridedGetWorkspaceSize") != nullptr;
}
namespace {
at::Tensor storage_span(const at::Tensor& state) {
  TORCH_CHECK(state.dim() >= 2 && state.size(0) > 0, "Invalid MegaGdnDecode state rank/slots");
  int64_t row = 1;
  for (int64_t d=state.dim()-1; d>0; --d) {
    TORCH_CHECK(state.stride(d) == row, "State inner dimensions must be contiguous");
    row *= state.size(d);
  }
  const int64_t stride = state.stride(0);
  TORCH_CHECK(stride >= row, "Overlapping MegaGdnDecode state slots");
  const int64_t max_elems=std::numeric_limits<int64_t>::max()/state.element_size();
  TORCH_CHECK(state.size(0)-1 <= (max_elems-row)/stride, "State address overflow");
  // as_strided preserves the original storage_offset and allocates no data.
  return state.as_strided({(state.size(0)-1)*stride+row}, {1}, state.storage_offset());
}
}
void mega_gdn_decode_strided(
    const at::Tensor& qkv,const at::Tensor& z,const at::Tensor& b,const at::Tensor& a,
    const at::Tensor& conv_weight,const at::Tensor& conv_state,const at::Tensor& a_log,
    const at::Tensor& dt_bias,const at::Tensor& ssm_state,const at::Tensor& read_indices,
    const at::Tensor& write_indices,const at::Tensor& norm_weight,const at::Tensor& conv_out,
    const at::Tensor& out,bool fla_ssm_state_layout) {
  TORCH_CHECK(qkv.dim()==2 && qkv.size(0)==1, "Strided MegaGdnDecode currently requires B=1");
  TORCH_CHECK(conv_state.dim()==3 && conv_state.size(1)==3 && conv_state.size(2)==qkv.size(1), "Invalid Conv state shape");
  TORCH_CHECK(ssm_state.dim()==4 && z.dim()==3 && z.size(0)==1 && z.size(2)==128 &&
              ssm_state.size(0)==conv_state.size(0) && ssm_state.size(1)==z.size(1) &&
              ssm_state.size(2)==128 && ssm_state.size(3)==128, "Invalid SSM state shape");
  for (const auto& t : {qkv,z,b,a,conv_weight,a_log,dt_bias,read_indices,write_indices,norm_weight,conv_out,out}) {
    TORCH_CHECK(t.is_contiguous() && t.device()==qkv.device(), "Strided MegaGdnDecode requires dense inputs on one device");
  }
  TORCH_CHECK(conv_state.device()==qkv.device() && ssm_state.device()==qkv.device(), "State device mismatch");
  TORCH_CHECK(conv_out.sizes()==qkv.sizes() && out.sizes()==z.sizes(), "Output shape mismatch");
  for (const auto& t : {qkv,z,b,a,conv_weight,conv_state,norm_weight,conv_out,out}) {
    TORCH_CHECK(t.scalar_type()==at::kBFloat16, "BF16 activations/state required");
  }
  for (const auto& t : {a_log,dt_bias,ssm_state}) {
    TORCH_CHECK(t.scalar_type()==at::kFloat, "FP32 recurrent state and decay weights required");
  }
  TORCH_CHECK(read_indices.scalar_type()==at::kInt && write_indices.scalar_type()==at::kInt, "INT32 indices required");
  auto flat_conv=storage_span(conv_state),flat_ssm=storage_span(ssm_state);
  int64_t slots=conv_state.size(0),conv_stride=conv_state.stride(0),ssm_stride=ssm_state.stride(0);
  EXEC_NPU_CMD(aclnnMegaGdnDecodeStrided,qkv,z,b,a,conv_weight,flat_conv,a_log,dt_bias,
               flat_ssm,read_indices,write_indices,norm_weight,slots,conv_stride,ssm_stride,
               fla_ssm_state_layout,conv_out,flat_conv,flat_ssm,out);
}
}
TORCH_LIBRARY_FRAGMENT(_C_ascend,m) {
  m.def("has_mega_gdn_decode_strided() -> bool");
  m.impl("has_mega_gdn_decode_strided",c10::DispatchKey::CompositeExplicitAutograd,
         &vllm_ascend::has_mega_gdn_decode_strided);
  m.def("mega_gdn_decode_strided(Tensor qkv, Tensor z, Tensor b, Tensor a, Tensor conv_weight, "
        "Tensor(a!) conv_state, Tensor a_log, Tensor dt_bias, Tensor(b!) ssm_state, "
        "Tensor read_indices, Tensor write_indices, Tensor norm_weight, "
        "Tensor(c!) conv_out, Tensor(d!) out, bool fla_ssm_state_layout=False) -> ()");
  m.impl("mega_gdn_decode_strided",torch::kPrivateUse1,&vllm_ascend::mega_gdn_decode_strided);
}
