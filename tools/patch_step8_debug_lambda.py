from __future__ import annotations

import re
from pathlib import Path

TARGET = Path("src/transformer_cit/run_ablation.py")

def die(msg: str) -> None:
    raise SystemExit(f"[PATCH-STEP8-DEBUG] ERROR: {msg}")

def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    if "PATCH_STEP8_DEBUG_LAMBDA_APPLIED" in text:
        print("[PATCH-STEP8-DEBUG] Already applied. Exiting.")
        return

    backup = TARGET.with_suffix(".py.bak_step8_debug_lambda")
    backup.write_text(text, encoding="utf-8")
    print(f"[PATCH-STEP8-DEBUG] Backup written to: {backup}")

    # Insert lambda_cit_eff right after weights = sched.get_loss_weights()
    pat_weights = r'(\s*weights\s*=\s*sched\.get_loss_weights\(\)\s*\n)'
    m = re.search(pat_weights, text)
    if not m:
        die("Could not find 'weights = sched.get_loss_weights()' line.")
    insert = (
        m.group(1)
        + "            # Step-8 debug: effective lambda_cit\n"
        + "            lambda_cit_eff = float(\n"
        + "                weights.get(\n"
        + "                    \"lambda_cit\",\n"
        + "                    weights.get(\"lambda_cit_forge\", getattr(sched.config, \"lambda_cit_forge\", 0.0)),\n"
        + "                )\n"
        + "            )\n"
    )
    text = re.sub(pat_weights, insert, text, count=1)

    # In the Step-8 update condition, replace weights.get(...) > 0.0 with lambda_cit_eff > 0.0 (if present)
    text = text.replace(
        'and weights.get("lambda_cit", weights.get("lambda_cit_forge", 0.0)) > 0.0',
        'and lambda_cit_eff > 0.0',
    )
    text = text.replace(
        'and weights.get("lambda_cit", 0.0) > 0.0',
        'and lambda_cit_eff > 0.0',
    )

    # Scale loss_c by lambda_cit_eff (robust)
    text = text.replace(
        'loss_c = L_cit(a_crit, critics) * weights.get("lambda_cit", weights.get("lambda_cit_forge", 0.0))',
        'loss_c = L_cit(a_crit, critics) * lambda_cit_eff',
    )
    text = text.replace(
        'loss_c = L_cit(a_crit, critics) * weights.get("lambda_cit", 0.0)',
        'loss_c = L_cit(a_crit, critics) * lambda_cit_eff',
    )

    # Add debug fields to row dict (insert after a_key_used if present; else after loss_cit)
    if '"a_key_used": a_key,' in text and '"lambda_cit_eff"' not in text:
        text = text.replace(
            '"a_key_used": a_key,',
            '"a_key_used": a_key,\n'
            '                "lambda_cit_eff": lambda_cit_eff,\n'
            '                "opt_heads_active": (opt_heads is not None),\n'
            '                "a_nat_none": (a_nat is None),\n'
        )
    elif '"loss_cit": float(loss_c.item()),' in text and '"lambda_cit_eff"' not in text:
        text = text.replace(
            '"loss_cit": float(loss_c.item()),',
            '"loss_cit": float(loss_c.item()),\n'
            '                "lambda_cit_eff": lambda_cit_eff,\n'
            '                "opt_heads_active": (opt_heads is not None),\n'
            '                "a_nat_none": (a_nat is None),\n'
        )

    text += "\n# PATCH_STEP8_DEBUG_LAMBDA_APPLIED\n"
    TARGET.write_text(text, encoding="utf-8")
    print("[PATCH-STEP8-DEBUG] Patched run_ablation.py")

if __name__ == "__main__":
    main()