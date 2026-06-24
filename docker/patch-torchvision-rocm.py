#!/usr/bin/env python3
"""Patch torchvision._meta_registrations for ROCm compatibility.

The torchvision ROCm wheel registers CUDA-native ops (torchvision::nms)
that don't exist on ROCm torch. Wrap the registration in try/except so
import torchvision succeeds on ROCm.
"""

import pathlib
import sys

# Find torchvision install location
for p in [
    pathlib.Path(
        "/opt/venv/lib/python3.12/site-packages/torchvision/_meta_registrations.py"
    ),
    pathlib.Path(
        "/opt/venv/lib/python3.10/site-packages/torchvision/_meta_registrations.py"
    ),
    pathlib.Path(
        "/usr/local/lib/python3.12/site-packages/torchvision/_meta_registrations.py"
    ),
    pathlib.Path(
        "/usr/local/lib/python3.10/site-packages/torchvision/_meta_registrations.py"
    ),
]:
    if p.exists():
        target = p
        break
else:
    # Try to find it via import
    import importlib.util

    spec = importlib.util.find_spec("torchvision._meta_registrations")
    if spec and spec.origin:
        target = pathlib.Path(spec.origin)
    else:
        print("ERROR: could not find torchvision._meta_registrations.py")
        sys.exit(1)

text = target.read_text()

old_block = (
    '@torch.library.register_fake("torchvision::nms")\n'
    "def meta_nms(dets, scores, iou_threshold):\n"
    '    torch._check(dets.dim() == 2, lambda: f"boxes should be a 2d tensor, got {dets.dim()}D")\n'
    '    torch._check(dets.size(1) == 4, lambda: f"boxes should have 4 elements in dimension 1, got {dets.size(1)}")\n'
    '    torch._check(scores.dim() == 1, lambda: f"scores should be a 1d tensor, got {scores.dim()}")\n'
    "    torch._check(\n"
    "        dets.size(0) == scores.size(0),\n"
    '        lambda: f"boxes and scores should have same number of elements in dimension 0, got {dets.size(0)} and {scores.size(0)}",\n'
    "    )\n"
    "    ctx = torch._custom_ops.get_ctx()\n"
    "    num_to_keep = ctx.create_unbacked_symint()\n"
    "    return dets.new_empty(num_to_keep, dtype=torch.long)"
)

new_block = (
    "try:\n"
    '    @torch.library.register_fake("torchvision::nms")\n'
    "    def meta_nms(dets, scores, iou_threshold):\n"
    '        torch._check(dets.dim() == 2, lambda: f"boxes should be a 2d tensor, got {dets.dim()}D")\n'
    '        torch._check(dets.size(1) == 4, lambda: f"boxes should have 4 elements in dimension 1, got {dets.size(1)}")\n'
    '        torch._check(scores.dim() == 1, lambda: f"scores should be a 1d tensor, got {scores.dim()}")\n'
    "        torch._check(\n"
    "            dets.size(0) == scores.size(0),\n"
    '            lambda: f"boxes and scores should have same number of elements in dimension 0, got {dets.size(0)} and {scores.size(0)}",\n'
    "        )\n"
    "        ctx = torch._custom_ops.get_ctx()\n"
    "        num_to_keep = ctx.create_unbacked_symint()\n"
    "        return dets.new_empty(num_to_keep, dtype=torch.long)\n"
    "except RuntimeError:\n"
    "    def meta_nms(*args, **kwargs):\n"
    "        return args[0].new_empty(0, dtype=torch.long)"
)

if old_block not in text:
    print(f"ERROR: pattern not found in {target}")
    print(
        "The torchvision version may have changed. Showing context around 'register_fake':"
    )
    idx = text.find("register_fake")
    if idx >= 0:
        print(text[max(0, idx - 100) : idx + 600])
    sys.exit(1)

text = text.replace(old_block, new_block, 1)
target.write_text(text)
print(f"Patched {target} for ROCm compatibility")
