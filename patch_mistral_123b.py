"""
patch_mistral_123b.py
=====================
Patches model.py and pipeline_gpu.py for Mistral-Large-123B on 2×A100:
  1. Adds AwqConfig(do_fuse=False) to bypass broken ExLlama kernels
  2. Adds device_map="auto" support for multi-GPU model splitting
  3. Ensures probe heads land on the correct output device

Run from repo root:  python patch_mistral_123b.py
"""
from pathlib import Path
import sys

MODEL_PY = Path("src/transformer_cit/model.py")
PIPELINE_PY = Path("pipeline_gpu.py")


def patch_model_py():
    """Three modifications to model.py."""
    if not MODEL_PY.exists():
        print(f"[ERR] {MODEL_PY} not found. Run from repo root.")
        return False

    code = MODEL_PY.read_text(encoding="utf-8")

    # Check if already patched
    if "device_map" in code and "AwqConfig" in code:
        print(f"[SKIP] {MODEL_PY} already patched.")
        return True

    changes = 0

    # --- Patch 1a: Add device_map parameter to __init__ ---
    # Find the __init__ signature and add device_map parameter
    old_init = "quantize_4bit: bool = False,"
    new_init = "quantize_4bit: bool = False,\n        device_map: str = None,"
    if old_init in code and "device_map: str" not in code:
        code = code.replace(old_init, new_init, 1)
        changes += 1
        print("  [1a] Added device_map parameter to __init__")

    # --- Patch 1b: Add device_map + AwqConfig to from_pretrained ---
    # Find the from_pretrained call and inject device_map + AwqConfig before it
    old_load = "self.backbone = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)"
    new_load = """# --- Multi-GPU + ExLlama bypass (Mistral 123B patch) ---
        if device_map and not quantize_4bit:
            load_kwargs["device_map"] = device_map
            print(f"[INFO] Loading with device_map={device_map}")
        # Bypass broken ExLlama kernels for AWQ models
        try:
            from transformers import AwqConfig
            load_kwargs["quantization_config"] = AwqConfig(do_fuse=False)
            print("[INFO] AwqConfig(do_fuse=False) set — ExLlama kernels bypassed")
        except ImportError:
            print("[WARN] AwqConfig not available — skipping ExLlama bypass")
        # --- End patch ---
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)"""
    if "AwqConfig" not in code:
        code = code.replace(old_load, new_load, 1)
        changes += 1
        print("  [1b] Added AwqConfig + device_map to from_pretrained")

    # --- Patch 1c: Add output device detection for multi-GPU ---
    # After backbone is loaded, detect the output device
    old_probe = "self.probe_heads = nn.ModuleList()"
    new_probe = """# Detect output device for multi-GPU setups
        self._output_device = "cpu"
        if device_map:
            params = list(self.backbone.parameters())
            self._output_device = params[-1].device
            print(f"[INFO] Output device (multi-GPU): {self._output_device}")
        self.probe_heads = nn.ModuleList()"""
    if "_output_device" not in code:
        code = code.replace(old_probe, new_probe, 1)
        changes += 1
        print("  [1c] Added output device detection for multi-GPU")

    if changes == 0:
        print(f"[WARN] No changes made to {MODEL_PY} — patterns not found.")
        print("       The code may have a different structure than expected.")
        print("       Please check model.py manually.")
        return False

    MODEL_PY.write_text(code, encoding="utf-8")
    print(f"  [OK] {MODEL_PY}: {changes} patches applied")
    return True


def patch_pipeline_py():
    """Two modifications to pipeline_gpu.py."""
    if not PIPELINE_PY.exists():
        print(f"[ERR] {PIPELINE_PY} not found. Run from repo root.")
        return False

    code = PIPELINE_PY.read_text(encoding="utf-8")

    if "device_map" in code and "_output_device" in code:
        print(f"[SKIP] {PIPELINE_PY} already patched.")
        return True

    changes = 0

    # --- Patch 2a: Read device_map from config and pass to CITModel ---
    # Find where CITModel is created and add device_map
    # Look for the pattern where model config is read
    old_q4 = 'q4 = bool(mcfg.get("quantize_4bit", False))'
    new_q4 = '''q4 = bool(mcfg.get("quantize_4bit", False))
    dmap = mcfg.get("device_map", None)'''
    if "dmap" not in code and old_q4 in code:
        code = code.replace(old_q4, new_q4, 1)
        changes += 1
        print("  [2a-i] Added device_map reading from config")

    # Now find the CITModel constructor call and add device_map
    old_cit = "quantize_4bit=q4,"
    new_cit = "quantize_4bit=q4,\n        device_map=dmap,"
    if "device_map=dmap" not in code and old_cit in code:
        code = code.replace(old_cit, new_cit, 1)
        changes += 1
        print("  [2a-ii] Added device_map to CITModel constructor")

    # --- Patch 2b: Multi-GPU device detection branch ---
    # Find the device detection block and add multi-GPU branch
    old_device = '    if q4:'
    new_device = '''    if dmap:
        device = str(model._output_device)
        for ph in model.probe_heads:
            ph.to(device)
        print(f"[INFO] Multi-GPU mode: probe heads on {device}")
    elif q4:'''
    if "Multi-GPU mode" not in code and old_device in code:
        code = code.replace(old_device, new_device, 1)
        changes += 1
        print("  [2b] Added multi-GPU device detection branch")

    if changes == 0:
        print(f"[WARN] No changes made to {PIPELINE_PY} — patterns not found.")
        print("       Please check pipeline_gpu.py manually.")
        return False

    PIPELINE_PY.write_text(code, encoding="utf-8")
    print(f"  [OK] {PIPELINE_PY}: {changes} patches applied")
    return True


def main():
    print("=" * 60)
    print("  Mistral-Large-123B Patch")
    print("  ExLlama bypass + Multi-GPU support")
    print("=" * 60)
    print()

    ok1 = patch_model_py()
    print()
    ok2 = patch_pipeline_py()
    print()

    if ok1 and ok2:
        print("[SUCCESS] All patches applied. Ready to run:")
        print("  python pipeline_gpu.py --config configs/mistral_123b_gpu_v2.yaml --seed 123")
    else:
        print("[PARTIAL] Some patches failed. Check output above.")
        print("You may need to apply changes manually.")

    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
