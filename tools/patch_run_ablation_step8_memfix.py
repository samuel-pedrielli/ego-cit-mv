from __future__ import annotations

import re
from pathlib import Path

TARGET = Path("src/transformer_cit/run_ablation.py")

def die(msg: str) -> None:
    raise SystemExit(f"[PATCH-STEP8-MEMFIX] ERROR: {msg}")

def main() -> None:
    if not TARGET.exists():
        die(f"Missing: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")

    if "PATCH_STEP8_MEMFIX_APPLIED" in text:
        print("[PATCH-STEP8-MEMFIX] Already applied. Exiting.")
        return

    backup = TARGET.with_suffix(".py.bak_step8_memfix")
    backup.write_text(text, encoding="utf-8")
    print(f"[PATCH-STEP8-MEMFIX] Backup written to: {backup}")

    # 1) Replace the model_nat deepcopy block with ref_heads_sd (heads-only reference)
    pat_nat = re.compile(
        r"\n\s*# Step-8: NAT model \(preserve target\) \+ heads-only optimizer\s*\n"
        r"\s*model_nat\s*=\s*copy\.deepcopy\(model\)\s*\n"
        r"\s*model_nat\.eval\(\)\s*\n"
        r"(?:\s*for p in model_nat\.parameters\(\):\s*\n\s*p\.requires_grad\s*=\s*False\s*\n)+"
        r"\s*\n"
        r"\s*# Trainable heads on TRAIN model \(backbone frozen\)\s*\n",
        re.M,
    )
    m = pat_nat.search(text)
    if not m:
        die("Could not find model_nat deepcopy block to replace (pattern mismatch).")

    repl = (
        "\n    # Step-8: reference (NAT) heads for preserve term (NO backbone clone)\n"
        "    ref_heads_sd = None\n"
        "    if hasattr(model, \"probe_heads\"):\n"
        "        ref_heads_sd = [\n"
        "            {k: v.detach().cpu().clone() for k, v in ph.state_dict().items()}\n"
        "            for ph in model.probe_heads\n"
        "        ]\n"
        "\n"
        "    # Trainable heads on TRAIN model (backbone frozen)\n"
    )
    text = pat_nat.sub(repl, text, count=1)

    # 2) Remove out_nat forward (no model_nat anymore)
    text = re.sub(
        r"\n\s*with torch\.no_grad\(\):\s*\n\s*out_nat\s*=\s*model_nat\(input_ids=input_ids,\s*attention_mask=attention_mask\)\s*\n",
        "\n",
        text,
        count=1,
        flags=re.M,
    )

    # 3) Replace 'a_nat = out_nat.get(...)' with heads-swap reference forward
    pat_anat = re.compile(r"^\s*a_nat\s*=\s*out_nat\.get\(a_key,\s*None\)\s*$", re.M)
    if not pat_anat.search(text):
        die("Could not find a_nat = out_nat.get(a_key, None) line to replace.")

    block = (
        "            a_nat = None\n"
        "            if ref_heads_sd is not None and hasattr(model, \"probe_heads\"):\n"
        "                # Save current heads\n"
        "                cur_sd = [\n"
        "                    {k: v.detach().cpu().clone() for k, v in ph.state_dict().items()}\n"
        "                    for ph in model.probe_heads\n"
        "                ]\n"
        "                n = min(len(model.probe_heads), len(ref_heads_sd))\n"
        "                # Load reference heads\n"
        "                for i in range(n):\n"
        "                    model.probe_heads[i].load_state_dict(ref_heads_sd[i])\n"
        "                # Forward with reference heads (backbone is already no_grad inside CITModel)\n"
        "                with torch.no_grad():\n"
        "                    out_ref = model(input_ids=input_ids, attention_mask=attention_mask)\n"
        "                a_nat = out_ref.get(a_key, None)\n"
        "                # Restore current heads\n"
        "                for i in range(n):\n"
        "                    model.probe_heads[i].load_state_dict(cur_sd[i])\n"
    )
    text = pat_anat.sub(block, text, count=1)

    text += "\n# PATCH_STEP8_MEMFIX_APPLIED\n"
    TARGET.write_text(text, encoding="utf-8")
    print(f"[PATCH-STEP8-MEMFIX] Patched: {TARGET}")
    print("[PATCH-STEP8-MEMFIX] Done.")

if __name__ == "__main__":
    main()