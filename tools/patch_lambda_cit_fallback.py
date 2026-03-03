from __future__ import annotations
from pathlib import Path

TARGET = Path("src/transformer_cit/run_ablation.py")

def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    if "lambda_cit_forge" in text and 'weights.get("lambda_cit", weights.get("lambda_cit_forge", 0.0))' in text:
        print("[PATCH-LAMBDA-CIT] Already applied. Exiting.")
        return

    backup = TARGET.with_suffix(".py.bak_lambda_cit")
    backup.write_text(text, encoding="utf-8")
    print(f"[PATCH-LAMBDA-CIT] Backup: {backup}")

    text2 = text.replace(
        'weights.get("lambda_cit", 0.0)',
        'weights.get("lambda_cit", weights.get("lambda_cit_forge", 0.0))'
    )

    if text2 == text:
        print("[PATCH-LAMBDA-CIT] Warning: no occurrences replaced (pattern not found).")
    else:
        TARGET.write_text(text2, encoding="utf-8")
        print("[PATCH-LAMBDA-CIT] Patched run_ablation.py")

if __name__ == "__main__":
    main()