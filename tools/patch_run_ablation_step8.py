from __future__ import annotations

import re
from pathlib import Path

TARGET = Path("src/transformer_cit/run_ablation.py")

def die(msg: str) -> None:
    raise SystemExit(f"[PATCH-STEP8] ERROR: {msg}")

def main() -> None:
    if not TARGET.exists():
        die(f"Missing file: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")

    if "lambda_preserve" in text and "cos_spread" in text and "critic_saturation" in text and "model_nat" in text:
        print("[PATCH-STEP8] Looks already patched. Exiting.")
        return

    backup = TARGET.with_suffix(".py.bak_step8")
    backup.write_text(text, encoding="utf-8")
    print(f"[PATCH-STEP8] Backup written to: {backup}")

    # ---------- 0) ensure needed imports ----------
    # Need: copy, math
    if "import copy" not in text:
        text = text.replace("import torch\n", "import torch\nimport copy\n")
    if "import math" not in text:
        # only if not present
        if "import math\n" not in text:
            text = text.replace("import copy\n", "import copy\nimport math\n")

    # ---------- 1) add helper functions (pairwise cosine spread) if absent ----------
    if "def _cos_spread" not in text:
        insert_point = text.find("def main()")
        if insert_point == -1:
            die("Could not find def main() to insert helpers before it.")
        helpers = (
            "\n\n"
            "def _cos_spread(x: torch.Tensor) -> float:\n"
            "    \"\"\"Mean pairwise cosine similarity in a batch. x: [B,d].\"\"\"\n"
            "    if x is None:\n"
            "        return float('nan')\n"
            "    B = int(x.shape[0])\n"
            "    if B < 2:\n"
            "        return float('nan')\n"
            "    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)\n"
            "    sim = x @ x.t()  # [B,B]\n"
            "    # exclude diagonal\n"
            "    return float(((sim.sum() - sim.diag().sum()) / (B * (B - 1))).item())\n"
        )
        text = text[:insert_point] + helpers + text[insert_point:]

    # ---------- 2) add ablation defaults parsing near tau ----------
    # Insert after tau line inside run_arm setup
    pat_tau = r"(\s*tau\s*=\s*float\(ab_cfg\.get\(\"ablation\".*?\)\s*# violation threshold.*\n)"
    m = re.search(pat_tau, text)
    if not m:
        die("Could not find tau assignment to insert Step8 params.")
    if "lambda_preserve" not in text:
        ins = m.group(1) + (
            "    # Step-8 CIT training params (safe defaults; can be overridden in ablation yaml)\n"
            "    cit_batch_size = int(ab_cfg.get(\"ablation\", {}).get(\"cit_batch_size\", 8))\n"
            "    lambda_preserve = float(ab_cfg.get(\"ablation\", {}).get(\"lambda_preserve\", 1.0))\n"
            "    sat_threshold = float(ab_cfg.get(\"ablation\", {}).get(\"sat_threshold\", 0.95))\n"
            "    stop_cos_spread = float(ab_cfg.get(\"ablation\", {}).get(\"stop_cos_spread\", 0.99))\n"
            "    stop_critic_saturation = float(ab_cfg.get(\"ablation\", {}).get(\"stop_critic_saturation\", 0.90))\n"
        )
        text = re.sub(pat_tau, ins, text, count=1)

    # ---------- 3) create model_nat clone after model creation and heads load ----------
    # Insert after heads load block end (after line with [WARN]/etc).
    if "model_nat" not in text:
        # find a stable anchor: after heads load block (pattern 'Optional: load trained probe heads')
        anchor = "# Optional: load trained probe heads (from train_heads.py)"
        idx = text.find(anchor)
        if idx == -1:
            die("Could not find heads load anchor comment.")
        # insert after the block that follows heads load; easiest: after the 'print([WARN] ...)' line.
        # We'll insert just after the heads load block by finding the next blank line after it.
        post = text.find("\n\n", idx)
        if post == -1:
            die("Could not locate insertion point after heads load block.")
        insert_at = post + 2

        block = (
            "    # Frozen NAT model for preserve term (same backbone+heads, no training)\n"
            "    model_nat = copy.deepcopy(model)\n"
            "    model_nat.eval()\n"
            "    for p in model_nat.parameters():\n"
            "        p.requires_grad = False\n"
            "\n"
            "    # Trainable model: only probe heads train (backbone stays frozen)\n"
            "    model.train()\n"
            "    for p in model.parameters():\n"
            "        p.requires_grad = False\n"
            "    if hasattr(model, \"probe_heads\"):\n"
            "        for ph in model.probe_heads:\n"
            "            for p in ph.parameters():\n"
            "                p.requires_grad = True\n"
            "    trainable_heads = [p for p in model.parameters() if p.requires_grad]\n"
            "    opt_heads = torch.optim.Adam(trainable_heads, lr=float(m_cfg.get(\"cit\", {}).get(\"lr_heads\", 1e-3))) if trainable_heads else None\n"
            "\n"
        )
        text = text[:insert_at] + block + text[insert_at:]

    # ---------- 4) change tokenization to batch for CIT arms ----------
    # Replace single-prompt tokenization with batch sampling block.
    # We locate the line: tok = model.tokenizer(task.prompt, return_tensors="pt")
    pat_tok = r"(\s*)tok\s*=\s*model\.tokenizer\(task\.prompt,\s*return_tensors=\"pt\"\)\s*\n\s*input_ids\s*=\s*tok\[\s*\"input_ids\"\s*\]\.to\(device\)\s*\n\s*attention_mask\s*=\s*tok\.get\(\"attention_mask\", torch\.ones_like\(input_ids\)\)\.to\(device\)\s*\n"
    m = re.search(pat_tok, text)
    if not m:
        die("Could not find single-prompt tokenization block to replace.")
    indent = m.group(1)
    batch_block = (
        f"{indent}# Build batch of prompts (for CIT training + guardrails)\n"
        f"{indent}# Sample with replacement from the promptpack\n"
        f"{indent}prompts_b = [tasks[(step + j) % len(tasks)].prompt for j in range(cit_batch_size)]\n"
        f"{indent}tok = model.tokenizer(\n"
        f"{indent}    prompts_b,\n"
        f"{indent}    return_tensors=\"pt\",\n"
        f"{indent}    padding=True,\n"
        f"{indent}    truncation=True,\n"
        f"{indent}    max_length=int(m_cfg.get(\"cit\", {{}}).get(\"max_length\", 256)),\n"
        f"{indent})\n"
        f"{indent}input_ids = tok[\"input_ids\"].to(device)\n"
        f"{indent}attention_mask = tok.get(\"attention_mask\", torch.ones_like(input_ids)).to(device)\n"
    )
    text = re.sub(pat_tok, batch_block, text, count=1)

    # ---------- 5) forward NAT + TRAIN and define a_nat/a_crit ----------
    # Replace: out = model(...)  # dict a1/a2/a3
    pat_out = r"(\s*)out\s*=\s*model\(input_ids=input_ids,\s*attention_mask=attention_mask\)\s*# dict a1/a2/a3\s*\n"
    m = re.search(pat_out, text)
    if not m:
        die("Could not find out=model(...) line to replace.")
    indent = m.group(1)
    out_block = (
        f"{indent}# Forward TRAIN model (probe heads may update)\n"
        f"{indent}out = model(input_ids=input_ids, attention_mask=attention_mask)  # dict a1/a2/a3\n"
        f"{indent}# Forward NAT model for preserve target (stop-grad)\n"
        f"{indent}with torch.no_grad():\n"
        f"{indent}    out_nat = model_nat(input_ids=input_ids, attention_mask=attention_mask)\n"
    )
    text = re.sub(pat_out, out_block, text, count=1)

    # Ensure we already have: a1 = out.get("a1", None) and a_crit = out.get(a_key,...)
    if "a_nat" not in text:
        # After a_crit line, insert a_nat
        pat_acrit = r"(a_crit\s*=\s*out\.get\(a_key,\s*None\)\s*\n)"
        m = re.search(pat_acrit, text)
        if not m:
            die("Could not find a_crit assignment to insert a_nat.")
        ins = m.group(1) + "            a_nat = out_nat.get(a_key, None)\n"
        text = re.sub(pat_acrit, ins, text, count=1)

    # ---------- 6) implement preserve + CIT updates (Forge only) ----------
    # Insert just before logging row dict, after loss computations block (before 'row = {')
    marker = "            row = {"
    idx = text.find(marker)
    if idx == -1:
        die("Could not find row logging dict start to insert Step8 training block.")
    # Insert once per file
    if "loss_preserve" not in text:
        insert = (
            "            # Step-8: CIT update on a_crit (a_key from critics ckpt, typically a3)\n"
            "            loss_preserve = torch.tensor(0.0, device=device)\n"
            "            cos_spread = None\n"
            "            critic_saturation = None\n"
            "            did_update = False\n"
            "            if opt_heads is not None and a_crit is not None and a_nat is not None:\n"
            "                # Guardrail metrics\n"
            "                cos_spread = _cos_spread(a_crit.detach())\n"
            "                with torch.no_grad():\n"
            "                    agg = critics(a_crit.detach())[\"aggregate\"]\n"
            "                    critic_saturation = float((agg > sat_threshold).float().mean().item())\n"
            "\n"
            "                # Hard stop if collapsing / saturating\n"
            "                if (cos_spread is not None and cos_spread > stop_cos_spread) or (\n"
            "                    critic_saturation is not None and critic_saturation > stop_critic_saturation\n"
            "                ):\n"
            "                    # disable further updates by dropping optimizer\n"
            "                    opt_heads = None\n"
            "                else:\n"
            "                    # Preserve term: keep close to NAT representation\n"
            "                    loss_preserve = lambda_preserve * (a_crit - a_nat.detach()).pow(2).sum(dim=-1).mean()\n"
            "\n"
            "                    # CIT loss: revision-then-pull (CITLoss already implements stop-grad target)\n"
            "                    # NOTE: we pass a_crit tensor regardless of name 'a1' in CITLoss.\n"
            "                    loss_c = L_cit(a_crit, critics) * weights.get(\"lambda_cit\", 0.0)\n"
            "\n"
            "                    # total update loss (only heads receive grad)\n"
            "                    loss_update = loss_c + loss_preserve\n"
            "                    opt_heads.zero_grad(set_to_none=True)\n"
            "                    loss_update.backward()\n"
            "                    opt_heads.step()\n"
            "                    did_update = True\n"
            "\n"
        )
        text = text[:idx] + insert + text[idx:]

    # ---------- 7) add new fields into row dict ----------
    # Insert new keys after 'loss_cit'
    if "\"cos_spread\"" not in text:
        text = text.replace(
            "\"loss_cit\": float(loss_c.item()),",
            "\"loss_cit\": float(loss_c.item()),\n"
            "                \"loss_preserve\": float(loss_preserve.item()) if loss_preserve is not None else None,\n"
            "                \"cos_spread\": cos_spread,\n"
            "                \"critic_saturation\": critic_saturation,\n"
            "                \"did_update\": did_update,\n"
            "                \"a_key_used\": a_key,\n"
        )

    TARGET.write_text(text, encoding="utf-8")
    print(f"[PATCH-STEP8] Patched: {TARGET}")
    print("[PATCH-STEP8] Done.")

if __name__ == "__main__":
    main()