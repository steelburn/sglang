#!/usr/bin/env python3
"""Create a stub sgl_kernel package for ROCm.

SGLang imports many symbols from sgl_kernel at module level. On ROCm, the
CUDA-only sglang-kernel wheel cannot be used. This stub provides all expected
symbols, raising RuntimeError if actually called (since these ops don't exist
on ROCm), but allowing module imports to succeed.

Most kernels have pure-Python/Triton fallbacks in SGLang, so the stub is
only reached for operations that are genuinely unavailable on ROCm.
"""

import shutil
import sys
from pathlib import Path

# Find site-packages
SITE_PACKAGES = None
for p in [
    Path("/opt/venv/lib/python3.12/site-packages"),
    Path("/opt/venv/lib/python3.10/site-packages"),
    Path("/usr/local/lib/python3.12/site-packages"),
    Path("/usr/local/lib/python3.10/site-packages"),
]:
    if p.exists():
        SITE_PACKAGES = p
        break

if not SITE_PACKAGES:
    print("ERROR: could not find site-packages")
    sys.exit(1)

STUB_DIR = SITE_PACKAGES / "sgl_kernel"

# Remove existing if it's our stub
if STUB_DIR.exists():
    if (STUB_DIR / "__init__.py").exists():
        content = (STUB_DIR / "__init__.py").read_text()
        if "STUB" in content:
            print(f"Removing existing stub at {STUB_DIR}")
            shutil.rmtree(STUB_DIR)
        else:
            print(f"ERROR: {STUB_DIR} exists and is not a stub, refusing to overwrite")
            sys.exit(1)

STUB_DIR.mkdir(parents=True, exist_ok=True)

# All symbols imported from sgl_kernel across the entire SGLang codebase.
# Each maps to a callable that raises RuntimeError if actually invoked.
SYMBOLS = [
    "apply_token_bitmask_inplace_cuda",
    "awq_dequantize",
    "bmm_fp8",
    "causal_conv1d_fwd",
    "causal_conv1d_update",
    "concat_mla_absorb_q",
    "concat_mla_k",
    "cutlass_mla_decode",
    "cutlass_mla_get_workspace_size",
    "cutlass_scaled_fp4_mm",
    "dsv3_fused_a_gemm",
    "dsv3_router_gemm",
    "es_sm100_mxfp8_blockscaled_grouped_mm",
    "es_sm100_mxfp8_blockscaled_grouped_quant",
    "fast_topk",
    "flash_mla_decode",
    "flash_mla_get_workspace_size",
    "fp8_blockwise_scaled_mm",
    "fp8_scaled_mm",
    "fused_add_rmsnorm",
    "fused_experts",
    "fused_qk_norm_rope",
    "gelu_and_mul",
    "gelu_quick",
    "gelu_tanh_and_mul",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "gptq_gemm",
    "gptq_shuffle",
    "hadamard_transform",
    "int8_scaled_mm",
    "kimi_k2_moe_fused_gate",
    "merge_state_v2",
    "moe_align_block_size",
    "moe_fused_gate",
    "moe_sum",
    "moe_sum_reduce",
    "qserve_w4a8_per_chn_gemm",
    "qserve_w4a8_per_group_gemm",
    "rmsnorm",
    "rotary_embedding",
    "scaled_fp4_quant",
    "segment_packbits",
    "sgl_per_token_group_quant_8bit",
    "sgl_per_token_group_quant_fp8",
    "sgl_per_token_group_quant_int8",
    "sgl_per_token_quant_fp8",
    "silu_and_mul",
    "spatial",
    "topk_sigmoid",
    "topk_softmax",
    "transfer_kv_all_layer",
    "transfer_kv_per_layer",
    "verify_tree_greedy",
    "weak_ref_tensor",
]

# Build __init__.py content
lines = [
    "# sgl_kernel package STUB for ROCm",
    "# Created by create-sgl-kernel-stub.py (Docker build)",
    "# Real kernels are intentionally not available - each raises RuntimeError",
    "",
    "import functools",
    "",
    "STUB = True",
    "",
]


def make_raiser(name):
    """Generate a stub function that raises RuntimeError when called."""
    return (
        f"def _{name}(*args, **kwargs):\n"
        f"    raise RuntimeError(\n"
        f'        f"sgl_kernel.{name} is not available on ROCm. "\n'
        f'        f"Install sglang-kernel built for ROCm or use fallback."\n'
        f"    )\n\n"
        f"{name} = _{name}\n"
    )


for sym in SYMBOLS:
    lines.append(make_raiser(sym))

(STUB_DIR / "__init__.py").write_text("\n".join(lines))

# Create subdirectories that are imported
for sub in [
    "kvcacheio",
    "flash_mla",
    "flash_attn",
    "speculative",
    "sparse_flash_attn",
    "allreduce",
    "elementwise",
    "flash_mla",
]:
    subdir = STUB_DIR / sub
    subdir.mkdir(exist_ok=True)
    (subdir / "__init__.py").write_text(
        f"# Stub for sgl_kernel.{sub} (ROCm)\n"
        f"# Real kernels not available on ROCm.\n"
        f"STUB = True\n"
    )

# Verify
sys.path.insert(0, str(STUB_DIR.parent))
import sgl_kernel  # type: ignore

print(f"sgl_kernel stub created at {STUB_DIR}")
print(f"  STUB = {getattr(sgl_kernel, 'STUB', False)}")
print(f"  Symbols provided: {len(SYMBOLS)}")
# Verify a few key symbols
for check in [
    "gelu_and_mul",
    "gelu_quick",
    "gelu_tanh_and_mul",
    "silu_and_mul",
    "fast_topk",
    "weak_ref_tensor",
]:
    print(f"    {check}: {hasattr(sgl_kernel, check)}")
