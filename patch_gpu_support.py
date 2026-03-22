"""
patch_gpu_support.py
====================
Adds GPU + 4-bit quantization support to run_ablation.py for Llama-3.1-70B.

Changes:
  1. Reads quantize_4bit and torch_dtype from model config
  2. Passes them to CITModel constructor
  3. Handles device placement for quantized models (device_map="auto")

Run from repo root:  python patch_gpu_support.py
"""
import re
from pathlib import Path

TARGET = Path("src/transformer_cit/run_ablation.py")

def apply():
    if not TARGET.exists():
        print(f"[ERR] {TARGET} not found. Run from repo root.")
        return False

    code = TARGET.read_text(encoding="utf-8")

    if "quantize_4bit" in code:
        print("[SKIP] GPU support patch already applied.")
        return True

    # --- Patch 1: Add quantize_4bit and torch_dtype reading after use_mlp_heads ---
    old_1 = '    use_mlp_heads = bool(m_cfg["model"].get("use_mlp_heads", False))'
    new_1 = '''    use_mlp_heads = bool(m_cfg["model"].get("use_mlp_heads", False))
    quantize_4bit = bool(m_cfg["model"].get("quantize_4bit", False))
    model_torch_dtype = str(m_cfg["model"].get("torch_dtype", "float32"))'''

    if old_1 not in code:
        print("[ERR] Cannot find use_mlp_heads line for Patch 1.")
        return False
    code = code.replace(old_1, new_1, 1)

    # --- Patch 2: Update CITModel creation to pass new params + handle device ---
    old_2 = '''    model = CITModel(
        model_name=model_name,
        tap_layers=tap_layers,
        d=d,
        pooling=pooling,
        use_mlp_heads=use_mlp_heads,
    ).to(device)'''

    new_2 = '''    model = CITModel(
        model_name=model_name,
        tap_layers=tap_layers,
        d=d,
        pooling=pooling,
        use_mlp_heads=use_mlp_heads,
        quantize_4bit=quantize_4bit,
        torch_dtype=model_torch_dtype,
    )
    if not quantize_4bit:
        model = model.to(device)
    else:
        # With device_map="auto", backbone is already on GPU.
        # Move probe_heads to the same device as the backbone output.
        gpu_dev = next(model.backbone.parameters()).device
        for ph in model.probe_heads:
            ph.to(gpu_dev)
        device = str(gpu_dev)  # override for critics, anchors, etc.
        print(f"[INFO] 4-bit mode: device overridden to {device}")'''

    if old_2 not in code:
        print("[ERR] Cannot find CITModel creation block for Patch 2.")
        return False
    code = code.replace(old_2, new_2, 1)

    TARGET.write_text(code, encoding="utf-8")
    print("[OK] GPU + 4-bit support patch applied to run_ablation.py")
    return True


if __name__ == "__main__":
    apply()
