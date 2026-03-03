from __future__ import annotations

from pathlib import Path

TARGET = Path("src/transformer_cit/run_ablation.py")


def die(msg: str) -> None:
    raise SystemExit(f"[PATCH-STEP8-V2] ERROR: {msg}")


def main() -> None:
    if not TARGET.exists():
        die(f"Missing: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")

    # Idempotence guard
    if "PATCH_STEP8_V2_APPLIED" in text:
        print("[PATCH-STEP8-V2] Already applied. Exiting.")
        return

    backup = TARGET.with_suffix(".py.bak_step8_v2")
    backup.write_text(text, encoding="utf-8")
    print(f"[PATCH-STEP8-V2] Backup written to: {backup}")

    # 0) import copy (needed for model_nat)
    if "import copy" not in text:
        marker = "import torch\n"
        if marker not in text:
            die("Could not find 'import torch' to insert import copy.")
        text = text.replace(marker, marker + "import copy\n", 1)

    # 1) helper _cos_spread (insert before def run_arm)
    if "def _cos_spread(" not in text:
        marker = "\ndef run_arm("
        if marker not in text:
            die("Could not find 'def run_arm(' to insert _cos_spread.")
        helper = (
            "\n\ndef _cos_spread(x: torch.Tensor) -> float:\n"
            "    \"\"\"Mean pairwise cosine similarity in a batch. x: [B,d].\"\"\"\n"
            "    if x is None:\n"
            "        return float('nan')\n"
            "    B = int(x.shape[0])\n"
            "    if B < 2:\n"
            "        return float('nan')\n"
            "    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)\n"
            "    sim = x @ x.t()  # [B,B]\n"
            "    return float(((sim.sum() - sim.diag().sum()) / (B * (B - 1))).item())\n"
        )
        text = text.replace(marker, helper + marker, 1)

    # 2) Step-8 params after tau assignment
    tau_line = '    tau = float(ab_cfg.get("ablation", {}).get("tau_welfare", 0.9))'
    if tau_line not in text:
        die("Could not find tau assignment line.")
    if "cit_batch_size" not in text:
        insert = (
            tau_line + "  # violation threshold (configurable)\n"
            "    # Step-8 CIT params (guardrails)\n"
            "    cit_batch_size = int(ab_cfg.get(\"ablation\", {}).get(\"cit_batch_size\", 8))\n"
            "    lambda_preserve = float(ab_cfg.get(\"ablation\", {}).get(\"lambda_preserve\", 1.0))\n"
            "    sat_threshold = float(ab_cfg.get(\"ablation\", {}).get(\"sat_threshold\", 0.95))\n"
            "    stop_cos_spread = float(ab_cfg.get(\"ablation\", {}).get(\"stop_cos_spread\", 0.99))\n"
            "    stop_critic_saturation = float(ab_cfg.get(\"ablation\", {}).get(\"stop_critic_saturation\", 0.90))\n"
        )
        text = text.replace(tau_line + "  # violation threshold (configurable)\n", insert, 1)

    # 3) Insert model_nat + heads-only optimizer before prev_a init
    prev_marker = "    prev_a: Dict[str, torch.Tensor] = {}\n"
    if prev_marker not in text:
        die("Could not find prev_a marker for insertion.")
    if "model_nat" not in text:
        block = (
            "    # Step-8: NAT model (preserve target) + heads-only optimizer\n"
            "    model_nat = copy.deepcopy(model)\n"
            "    model_nat.eval()\n"
            "    for p in model_nat.parameters():\n"
            "        p.requires_grad = False\n"
            "\n"
            "    # Trainable heads on TRAIN model (backbone frozen)\n"
            "    model.eval()\n"
            "    for p in model.parameters():\n"
            "        p.requires_grad = False\n"
            "    if hasattr(model, \"probe_heads\"):\n"
            "        for ph in model.probe_heads:\n"
            "            for p in ph.parameters():\n"
            "                p.requires_grad = True\n"
            "    trainable_heads = [p for p in model.parameters() if p.requires_grad]\n"
            "    opt_heads = (\n"
            "        torch.optim.Adam(\n"
            "            trainable_heads,\n"
            "            lr=float(m_cfg.get(\"cit\", {}).get(\"lr_heads\", 1e-3)),\n"
            "        )\n"
            "        if trainable_heads\n"
            "        else None\n"
            "    )\n"
            "\n"
        )
        text = text.replace(prev_marker, block + prev_marker, 1)

    # 4) Replace single-prompt tokenization + forward with batching + out_nat
    old_tok = (
        "            # Tokenize + forward (B=1)\n"
        "            tok = model.tokenizer(task.prompt, return_tensors=\"pt\")\n"
        "            input_ids = tok[\"input_ids\"].to(device)\n"
        "            attention_mask = tok.get(\"attention_mask\", torch.ones_like(input_ids)).to(device)\n"
        "\n"
        "            out = model(input_ids=input_ids, attention_mask=attention_mask)  # dict a1/a2/a3\n"
        "            a1 = out.get(\"a1\", None)\n"
    )
    if old_tok not in text:
        die("Could not find the expected B=1 tokenization block to replace.")
    new_tok = (
        "            # Tokenize + forward (B=cit_batch_size)\n"
        "            prompts_b = [task.prompt] + [tasks[(step + j) % len(tasks)].prompt for j in range(1, cit_batch_size)]\n"
        "            tok = model.tokenizer(\n"
        "                prompts_b,\n"
        "                return_tensors=\"pt\",\n"
        "                padding=True,\n"
        "                truncation=True,\n"
        "                max_length=int(m_cfg.get(\"cit\", {}).get(\"max_length\", 256)),\n"
        "            )\n"
        "            input_ids = tok[\"input_ids\"].to(device)\n"
        "            attention_mask = tok.get(\"attention_mask\", torch.ones_like(input_ids)).to(device)\n"
        "\n"
        "            out = model(input_ids=input_ids, attention_mask=attention_mask)  # dict a1/a2/a3\n"
        "            with torch.no_grad():\n"
        "                out_nat = model_nat(input_ids=input_ids, attention_mask=attention_mask)\n"
        "            a1 = out.get(\"a1\", None)\n"
    )
    text = text.replace(old_tok, new_tok, 1)

    # 5) Add a_nat after a_crit line
    line_acrit = "            a_crit = out.get(a_key, None)\n"
    if line_acrit not in text:
        die("Could not find a_crit line to extend with a_nat.")
    text = text.replace(line_acrit, line_acrit + "            a_nat = out_nat.get(a_key, None)\n", 1)

    # 6) Init guardrail/loss vars after loss_total init
    lt = "            loss_total = torch.tensor(0.0, device=device)\n"
    if lt not in text:
        die("Could not find loss_total init line.")
    if "loss_preserve" not in text:
        text = text.replace(
            lt,
            lt
            + "            loss_preserve = torch.tensor(0.0, device=device)\n"
            + "            cos_spread = None\n"
            + "            critic_saturation = None\n"
            + "            did_update = False\n",
            1,
        )

    # 7) Replace L_CIT block with heads-update + guardrails (revision-then-pull already in CITLoss)
    old_cit = (
        "             if \"L_CIT\" in losses_enabled and a_crit is not None:\n"
        "                 loss_c = L_cit(a_crit, critics) * weights.get(\"lambda_cit\", 0.0)\n"
        "                 loss_total = loss_total + loss_c\n"
        "             else:\n"
        "                 loss_c = torch.tensor(0.0, device=device)\n"
    )
    if old_cit not in text:
        die("Could not find the expected L_CIT block to replace.")
    new_cit = (
        "             if (\n"
        "                 \"L_CIT\" in losses_enabled\n"
        "                 and a_crit is not None\n"
        "                 and a_nat is not None\n"
        "                 and opt_heads is not None\n"
        "                 and weights.get(\"lambda_cit\", 0.0) > 0.0\n"
        "             ):\n"
        "                 cos_spread = _cos_spread(a_crit.detach())\n"
        "                 with torch.no_grad():\n"
        "                     agg_now = critics(a_crit.detach())[\"aggregate\"]\n"
        "                     critic_saturation = float((agg_now > sat_threshold).float().mean().item())\n"
        "\n"
        "                 if (cos_spread is not None and cos_spread > stop_cos_spread) or (\n"
        "                     critic_saturation is not None and critic_saturation > stop_critic_saturation\n"
        "                 ):\n"
        "                     opt_heads = None\n"
        "                     loss_c = torch.tensor(0.0, device=device)\n"
        "                 else:\n"
        "                     loss_c = L_cit(a_crit, critics) * weights.get(\"lambda_cit\", 0.0)\n"
        "                     loss_preserve = lambda_preserve * (a_crit - a_nat.detach()).pow(2).sum(dim=-1).mean()\n"
        "                     loss_update = loss_c + loss_preserve\n"
        "                     opt_heads.zero_grad(set_to_none=True)\n"
        "                     loss_update.backward()\n"
        "                     opt_heads.step()\n"
        "                     did_update = True\n"
        "\n"
        "                 loss_total = loss_total + loss_c + loss_preserve\n"
        "             else:\n"
        "                 loss_c = torch.tensor(0.0, device=device)\n"
    )
    text = text.replace(old_cit, new_cit, 1)

    # 8) Add new fields to row dict after loss_cit
    key_line = "                 \"loss_cit\": float(loss_c.item()),\n"
    if key_line not in text:
        die("Could not find loss_cit line to extend row dict.")
    if "\"cos_spread\"" not in text:
        text = text.replace(
            key_line,
            key_line
            + "                 \"loss_preserve\": float(loss_preserve.item()),\n"
            + "                 \"cos_spread\": cos_spread,\n"
            + "                 \"critic_saturation\": critic_saturation,\n"
            + "                 \"did_update\": did_update,\n"
            + "                 \"a_key_used\": a_key,\n",
            1,
        )

    # Tag
    text = text + "\n# PATCH_STEP8_V2_APPLIED\n"

    TARGET.write_text(text, encoding="utf-8")
    print(f"[PATCH-STEP8-V2] Patched: {TARGET}")
    print("[PATCH-STEP8-V2] Done.")


if __name__ == "__main__":
    main()