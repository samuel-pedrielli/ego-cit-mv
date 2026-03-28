#!/usr/bin/env python3
"""
CIT v2 — Mistral-Large-123B Rerun Patch (batch_size=2, 2×A100)
================================================================
Applies ALL modifications needed for the rerun in one shot.
No incremental patches. No manual sed fixes.

Changes:
  1. pipeline_gpu.py: Replace single-GPU force code with accelerate
     multi-GPU dispatch (CPU → 2×A100 split)
  2. configs/mistral_123b_gpu_v2.yaml: batch_size 1 → 2

Usage:
  python patch_rerun_bs2.py
"""

import re
import sys
import os

CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def patch_file(filepath, patches, label):
    """Apply a list of (description, old_pattern, new_text) patches to a file."""
    if not os.path.exists(filepath):
        print(f"{RED}[FAIL] {filepath} not found!{RESET}")
        return False

    with open(filepath, "r") as f:
        content = f.read()

    original = content
    applied = 0

    for desc, pattern, replacement in patches:
        if isinstance(pattern, str):
            # Literal string replacement
            if pattern in content:
                content = content.replace(pattern, replacement, 1)
                print(f"  {GREEN}[OK]{RESET} {desc}")
                applied += 1
            else:
                # Check if replacement is already present (idempotent)
                if replacement in content:
                    print(f"  {YELLOW}[SKIP]{RESET} {desc} — already applied")
                    applied += 1
                else:
                    print(f"  {RED}[MISS]{RESET} {desc} — pattern not found")
                    print(f"         Looking for: {pattern[:80]}...")
        else:
            # Regex replacement
            new_content, n = pattern.subn(replacement, content)
            if n > 0:
                content = new_content
                print(f"  {GREEN}[OK]{RESET} {desc}")
                applied += 1
            else:
                if re.search(re.escape(replacement[:60]) if isinstance(replacement, str) else '', content):
                    print(f"  {YELLOW}[SKIP]{RESET} {desc} — already applied")
                    applied += 1
                else:
                    print(f"  {RED}[MISS]{RESET} {desc} — regex not matched")

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)

    total = len(patches)
    print(f"  {GREEN if applied == total else RED}[{label}] {applied}/{total} patches applied{RESET}")
    return applied == total


def main():
    print(f"""
{CYAN}{'='*60}
  CIT v2 — Mistral Rerun Patch (batch_size=2, 2×A100)
  Robust: CPU load → accelerate dispatch → multi-GPU
{'='*60}{RESET}
""")

    all_ok = True

    # ──────────────────────────────────────────────────────────
    # PATCH 1: pipeline_gpu.py — multi-GPU dispatch via accelerate
    # ──────────────────────────────────────────────────────────
    print(f"{CYAN}[1] Patching pipeline_gpu.py — accelerate multi-GPU dispatch{RESET}")

    # We need to replace the single-GPU force block with accelerate dispatch.
    # The current code (after previous patches) has something like:
    #
    #   cur_dev = str(next(model.backbone.parameters()).device)
    #   if cur_dev == "cpu":
    #       model.backbone = model.backbone.to("cuda:0")
    #   device = str(next(model.backbone.parameters()).device)
    #   for ph in model.probe_heads:
    #       ph.to(device)
    #
    # We replace this entire block with accelerate dispatch logic.

    PIPELINE = "pipeline_gpu.py"

    # Read file to check what patterns exist
    if not os.path.exists(PIPELINE):
        print(f"  {RED}[FAIL] {PIPELINE} not found!{RESET}")
        sys.exit(1)

    with open(PIPELINE, "r") as f:
        pipeline_content = f.read()

    # Strategy: find the force-GPU block and replace it entirely
    # Pattern: from 'cur_dev = str(next(model.backbone' to 'ph.to(device)'
    # This handles both the exact code and minor variations

    # First, check if accelerate dispatch is already in place
    if "dispatch_model" in pipeline_content:
        print(f"  {YELLOW}[SKIP]{RESET} accelerate dispatch already present")
    else:
        # Try to find the force-GPU block with a flexible regex
        force_gpu_pattern = re.compile(
            r'([ \t]*)cur_dev\s*=\s*str\(next\(model\.backbone\.parameters\(\)\)\.device\).*?'
            r'for\s+ph\s+in\s+model\.probe_heads:\s*\n\s+ph\.to\(device\)',
            re.DOTALL
        )

        match = force_gpu_pattern.search(pipeline_content)
        if match:
            indent = match.group(1)
            replacement = f"""{indent}# === Multi-GPU dispatch via accelerate (rerun patch) ===
{indent}import torch as _torch_gpu_check
{indent}cur_dev = str(next(model.backbone.parameters()).device)
{indent}n_gpus = _torch_gpu_check.cuda.device_count()
{indent}print(f"[INFO] Model on {{cur_dev}}, GPUs available: {{n_gpus}}")
{indent}if cur_dev == "cpu" and n_gpus >= 2:
{indent}    from accelerate import dispatch_model, infer_auto_device_map
{indent}    max_mem = {{i: "70GiB" for i in range(n_gpus)}}
{indent}    dmap_auto = infer_auto_device_map(model.backbone, max_memory=max_mem)
{indent}    model.backbone = dispatch_model(model.backbone, dmap_auto)
{indent}    # Output device = device of last parameter (last layer)
{indent}    device = str(list(model.backbone.parameters())[-1].device)
{indent}    print(f"[INFO] Multi-GPU dispatch: {{n_gpus}} GPUs, output device: {{device}}")
{indent}elif cur_dev == "cpu":
{indent}    model.backbone = model.backbone.to("cuda:0")
{indent}    device = "cuda:0"
{indent}    print("[INFO] Single-GPU: moved to cuda:0")
{indent}else:
{indent}    device = cur_dev
{indent}    print(f"[INFO] Model already on {{device}}")
{indent}for ph in model.probe_heads:
{indent}    ph.to(device)"""

            pipeline_content = pipeline_content[:match.start()] + replacement + pipeline_content[match.end():]

            with open(PIPELINE, "w") as f:
                f.write(pipeline_content)
            print(f"  {GREEN}[OK]{RESET} Replaced force-GPU block with accelerate dispatch")
        else:
            # Fallback: try to find just the backbone.to("cuda:0") line
            simple_pattern = re.compile(
                r'([ \t]*)model\.backbone\s*=\s*model\.backbone\.to\("cuda:0"\)'
            )
            match2 = simple_pattern.search(pipeline_content)
            if match2:
                print(f"  {YELLOW}[WARN]{RESET} Found simple .to('cuda:0') but not full block")
                print(f"  {RED}[ACTION]{RESET} Send pipeline_gpu.py content to Claude for manual patch")
                all_ok = False
            else:
                print(f"  {RED}[MISS]{RESET} Force-GPU block not found — unknown code structure")
                print(f"  {RED}[ACTION]{RESET} Send pipeline_gpu.py content to Claude for manual patch")
                all_ok = False

    # Also ensure 'accelerate' import is at top level (for safety)
    with open(PIPELINE, "r") as f:
        pipeline_content = f.read()

    if "from accelerate import" not in pipeline_content and "import accelerate" not in pipeline_content:
        # Add a safety import comment (actual import is inline in the dispatch block)
        pass  # Inline import is fine — no top-level needed

    # ──────────────────────────────────────────────────────────
    # PATCH 2: Config — batch_size=2
    # ──────────────────────────────────────────────────────────
    print(f"\n{CYAN}[2] Patching configs/mistral_123b_gpu_v2.yaml — batch_size=2{RESET}")

    CONFIG = "configs/mistral_123b_gpu_v2.yaml"
    if not os.path.exists(CONFIG):
        print(f"  {RED}[FAIL] {CONFIG} not found!{RESET}")
        all_ok = False
    else:
        with open(CONFIG, "r") as f:
            cfg = f.read()

        # Change batch_size: 1 → 2
        if "batch_size: 1" in cfg:
            cfg = cfg.replace("batch_size: 1", "batch_size: 2", 1)
            with open(CONFIG, "w") as f:
                f.write(cfg)
            print(f"  {GREEN}[OK]{RESET} batch_size: 1 → 2")
        elif "batch_size: 2" in cfg:
            print(f"  {YELLOW}[SKIP]{RESET} batch_size already set to 2")
        else:
            print(f"  {RED}[MISS]{RESET} batch_size field not found in config")
            all_ok = False

    # ──────────────────────────────────────────────────────────
    # VALIDATION
    # ──────────────────────────────────────────────────────────
    print(f"\n{CYAN}[3] Validation{RESET}")

    # Re-read pipeline to validate
    with open(PIPELINE, "r") as f:
        final = f.read()

    checks = [
        ("dispatch_model in pipeline", "dispatch_model" in final),
        ("infer_auto_device_map in pipeline", "infer_auto_device_map" in final),
        ("probe heads moved to device", "ph.to(device)" in final),
    ]

    # Validate config
    with open(CONFIG, "r") as f:
        cfg_final = f.read()
    checks.append(("batch_size: 2 in config", "batch_size: 2" in cfg_final))

    for desc, ok in checks:
        status = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {status} {desc}")
        if not ok:
            all_ok = False

    # ──────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────
    print()
    if all_ok:
        print(f"""{GREEN}{'='*60}
  [SUCCESS] All patches applied and validated.
  
  Ready to run:
    python pipeline_gpu.py --config configs/mistral_123b_gpu_v2.yaml --seed 123
    
  Expected behavior:
    - Model loads on CPU via gptqmodel
    - accelerate dispatches across 2×A100
    - batch_size=2 (identical to Qwen runs)
    - Results saved to results/v2/seed123/
{'='*60}{RESET}""")
    else:
        print(f"""{RED}{'='*60}
  [INCOMPLETE] Some patches failed.
  Send the output above + file contents to Claude.
{'='*60}{RESET}""")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
