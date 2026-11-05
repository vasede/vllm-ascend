# MegaGdnDecode for non-speculative Qwen decode

Set `additional_config.ascend_fusion_config.fusion_ops_mega_gdn_decode=true`
to use the xLLM MegaGdnDecode ACLNN operator on Ascend 950. The default is false.
The implementation targets pure decode batches with speculative decoding disabled.
Prefill, mixed batches, and unsupported geometries continue through the existing GDN path.

```bash
source /path/to/mega_gdn_opp/vendors/custom_mega_gdn_xllm_math/bin/set_env.bash
vllm serve MODEL --dtype bfloat16 --mamba-ssm-cache-dtype float32 \
  --additional-config '{"ascend_fusion_config":{"fusion_ops_mega_gdn_decode":true}}'
```

Merge the fusion field into an existing additional-config object if one is already in use.
Do not configure speculative decoding for this path.

The fusion includes causal convolution and SiLU, Q/K L2 normalization, gate
calculation, recurrent state update, and gated RMSNorm. Input and output projections
remain outside the operator. The framework passes `z` into a mutating custom-op
boundary and skips the external norm only when the fusion actually ran. KV connector
wait/save hooks and output-projection prefetch ordering are preserved.

Required geometry and precision:

- One token per live decode request; batch or captured batch size at most 32.
- Key and value head dimensions 128; convolution width 4, no convolution bias.
- Local key heads in {1, 2, 4, 8, 16}; value/key head ratio an integer from 1 to 4.
- BF16 activations, convolution cache and norm weights; FP32 recurrent state.
- SiLU activation and output gate; RMSNorm epsilon 1e-6 and norm before gate.
- Conv cache uses SD layout; recurrent cache uses the existing vLLM `[V,K]` layout.

Continuous state pools of at most 1024 slots are passed directly. For block-strided
state pools or larger pools, a paired Triton gather collects only this batch's
Conv/SSM rows, MegaGdnDecode updates the compact buffers, and a paired scatter writes
back live rows. This adds two device kernels. Inner state rows must be contiguous.
The compact path ignores reserved padding slot zero and preserves physical page gaps.
No device index values are read back to the CPU.

Build the A5 OPP from xLLM-AI/xllm-ops commit
`d5694d2248e24902b5a6331aa909b283a38f5bd6` and rebuild `vllm_ascend_C` to register
`torch.ops._C_ascend.mega_gdn_decode`. A separate OPP prefix avoids replacing an
existing vendor installation. From the xllm-ops checkout after initializing PTO:

```bash
git submodule update --init --depth 1 third_party/pto-isa
cd xllm_ops
bash build.sh --pkg --ops=mega_gdn_decode --soc=ascend950 \
  --vendor_name=custom_mega_gdn -j8
bash build/cann-ops-xllm-custom_mega_gdn_linux-x86_64.run \
  --install-path=/path/to/mega_gdn_opp
```

Use the Python/CANN environment of the target worker for building and installation.
The OPP environment must be sourced before the worker imports the C++ extension.

Verification commands from the vllm-ascend checkout:

```bash
python -m pytest -q tests/ut/ops/test_mega_gdn_decode.py tests/ut/ops/test_gdn.py
python -m pytest -q -s \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_mega_gdn_decode.py
python benchmarks/benchmark_mega_gdn_decode.py --batch 4 --key-heads 4 --value-heads 8
```

The NPU tests cover output and both state caches, contiguous and strided pools,
padding ownership, graph replay with changing inputs/indices, and twelve consecutive
steps against the existing FLA decode core. Floating-point rounding differs from
the original chain, so this is an opt-in integration rather than a bitwise-equivalent
replacement. Validate model-level accuracy and throughput with production weights
before choosing it for a workload.
