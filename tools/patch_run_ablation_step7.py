from __future__ import annotations

import re
from pathlib import Path

TARGET = Path("src/transformer_cit/run_ablation.py")

def die(msg: str) -> None:
    raise SystemExit(f"[PATCH-STEP7] ERROR: {msg}")

def main() -> None:
    if not TARGET.exists():
        die(f"Missing file: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")
    if "--critics" in text and "critics_path" in text and "a_key" in text and 'out.get(args.a_key' in text:
        print("[PATCH-STEP7] Looks already patched. Exiting.")
        return

    backup = TARGET.with_suffix(".py.bak_step7")
    backup.write_text(text, encoding="utf-8")
    print(f"[PATCH-STEP7] Backup written to: {backup}")

    # 1) Add --critics CLI arg after --heads
    if "--critics" not in text:
        lines = text.splitlines(True)
        out_lines = []
        inserted = False
        for ln in lines:
            out_lines.append(ln)
            if (not inserted) and ('ap.add_argument("--heads"' in ln):
                out_lines.append('    ap.add_argument("--critics", default="", help="Path to critics checkpoint (.pt) produced by train_critics.py")\n')
                inserted = True
        if not inserted:
            die("Could not find argparse --heads line to insert --critics.")
        text = "".join(out_lines)

    # 2) Add critics_path parsing after heads_path parsing
    if "critics_path" not in text:
        pat = r"(heads_path\s*=\s*Path\(args\.heads\)\s*if\s*args\.heads\s*else\s*None\s*\n)"
        m = re.search(pat, text)
        if not m:
            die("Could not find heads_path parsing line.")
        ins = m.group(1) + "    critics_path = Path(args.critics) if args.critics else None\n"
        text = re.sub(pat, ins, text, count=1)

    # 3) Add critics_path param in run_arm signature (after heads_path)
    if re.search(r"^\s*critics_path:\s*Optional\[Path\],\s*$", text, flags=re.M) is None:
        pat = r"(^\s*heads_path:\s*Optional\[Path\],\s*\n)"
        m = re.search(pat, text, flags=re.M)
        if not m:
            die("Could not find 'heads_path: Optional[Path],' in run_arm signature.")
        ins = m.group(1) + "    critics_path: Optional[Path],\n"
        text = re.sub(pat, ins, text, count=1, flags=re.M)

    # 4) Pass critics_path into run_arm call in main()
    pat_call = r"(run_arm\(.+?heads_path\s*=\s*heads_path,\s*)(dry\s*=\s*args\.dry\))"
    if re.search(pat_call, text, flags=re.S) is None:
        # fallback: match the exact call style seen in repo
        pat_call2 = r"(run_arm\([^)]*heads_path=heads_path,\s*)(dry=args\.dry\))"
        if re.search(pat_call2, text) is None:
            die("Could not find run_arm(...) call with heads_path=... to patch.")
        text = re.sub(pat_call2, r"\1critics_path=critics_path, \2", text, count=1)
    else:
        text = re.sub(pat_call, r"\1critics_path=critics_path, \2", text, count=1, flags=re.S)

    # 5) Replace critics initialization block (random -> optional load calibrated)
    # Original block expected:
    #     # Critics
    #     K = int(...)
    #     critics = CriticEnsemble(K=K, d=d)
    #     critics.freeze()
    pat_crit = (
        r"(?P<indent>^[ \t]*)# Critics\s*\n"
        r"(?P=indent)K\s*=\s*int\(m_cfg\.get\(\"critics\".*?\)\)\s*\n"
        r"(?P=indent)critics\s*=\s*CriticEnsemble\(K=K,\s*d=d\)\s*\n"
        r"(?P=indent)critics\.freeze\(\)\s*\n"
    )
    m = re.search(pat_crit, text, flags=re.M | re.S)
    if not m:
        die("Could not find critics init block to replace.")
    indent = m.group("indent")

    new_block = (
        f"{indent}# Critics\n"
        f"{indent}K = int(m_cfg.get(\"critics\", {{}}).get(\"num_rules\", 5))\n"
        f"{indent}critics = CriticEnsemble(K=K, d=d).to(device)\n"
        f"{indent}a_key = \"a1\"\n"
        f"{indent}\n"
        f"{indent}# Optional: load calibrated critics checkpoint (from train_critics.py)\n"
        f"{indent}if critics_path is not None and critics_path.exists():\n"
        f"{indent}    ckp = torch.load(str(critics_path), map_location=device)\n"
        f"{indent}    sd = ckp.get(\"state_dict\", None)\n"
        f"{indent}    meta = ckp.get(\"meta\", {{}}) or {{}}\n"
        f"{indent}    rule_names = ckp.get(\"rule_names\", None)\n"
        f"{indent}    if isinstance(rule_names, list) and len(rule_names) > 0:\n"
        f"{indent}        K = len(rule_names)\n"
        f"{indent}        critics = CriticEnsemble(K=K, d=d).to(device)\n"
        f"{indent}    if sd is None:\n"
        f"{indent}        raise RuntimeError(f\"Critics checkpoint missing state_dict: {{critics_path}}\")\n"
        f"{indent}    critics.load_state_dict(sd)\n"
        f"{indent}    a_key = str(meta.get(\"a_key\", \"a1\"))\n"
        f"{indent}    if a_key not in (\"a1\", \"a2\", \"a3\"):\n"
        f"{indent}        a_key = \"a1\"\n"
        f"{indent}    print(f\"[INFO] Loaded critics from: {{critics_path}} (a_key={{a_key}}, K={{K}}, d={{d}})\")\n"
        f"{indent}\n"
        f"{indent}critics.freeze()\n"
        f"{indent}critics.eval()\n"
    )
    text = re.sub(pat_crit, new_block, text, count=1, flags=re.M | re.S)

    # 6) Insert a_crit extraction after a1
    if "a_crit" not in text:
        pat_a1 = r"(a1\s*=\s*out\.get\(\"a1\",\s*None\)\s*\n)"
        m = re.search(pat_a1, text)
        if not m:
            die("Could not find a1 extraction line to add a_crit.")
        ins = m.group(1) + "            a_crit = out.get(a_key, None)\n"
        text = re.sub(pat_a1, ins, text, count=1)

    # 7) Replace welfare proxy block to use a_crit
    pat_w = (
        r"(?P<indent>^[ \t]*)# Welfare proxy: use critic aggregate as w_t in \[0,1\]\s*\n"
        r"(?P=indent)welfare\s*=\s*None\s*\n"
        r"(?P=indent)violation\s*=\s*None\s*\n"
        r"(?P=indent)if\s+a1\s+is\s+not\s+None:\s*\n"
        r"(?P=indent)[ \t]+welfare\s*=\s*float\(critics\(a1\)\[\"aggregate\"\]\.mean\(\)\.item\(\)\)\s*\n"
        r"(?P=indent)[ \t]+violation\s*=\s*1\.0\s*if\s*welfare\s*<\s*tau\s*else\s*0\.0\s*\n"
        r"(?P=indent)[ \t]+viol_hist\.append\(float\(violation\)\)\s*\n"
    )
    m = re.search(pat_w, text, flags=re.M)
    if not m:
        die("Could not find welfare proxy block to replace.")
    indent = m.group("indent")
    new_w = (
        f"{indent}# Welfare proxy: use critic aggregate as w_t in [0,1]\n"
        f"{indent}welfare = None\n"
        f"{indent}violation = None\n"
        f"{indent}if a_crit is not None:\n"
        f"{indent}    welfare = float(critics(a_crit)[\"aggregate\"].mean().item())\n"
        f"{indent}    violation = 1.0 if welfare < tau else 0.0\n"
        f"{indent}    viol_hist.append(float(violation))\n"
    )
    text = re.sub(pat_w, new_w, text, count=1, flags=re.M)

    # 8) Update L_welfare and L_CIT to use a_crit (monitor-only for now)
    text = text.replace('if "L_welfare" in losses_enabled and a1 is not None:',
                        'if "L_welfare" in losses_enabled and a_crit is not None:')
    text = text.replace('cw = critics(a1)["aggregate"].unsqueeze(-1)',
                        'cw = critics(a_crit)["aggregate"].unsqueeze(-1)')
    text = text.replace('if "L_CIT" in losses_enabled and a1 is not None:',
                        'if "L_CIT" in losses_enabled and a_crit is not None:')
    text = text.replace('loss_c = L_cit(a1, critics)',
                        'loss_c = L_cit(a_crit, critics)')

    TARGET.write_text(text, encoding="utf-8")
    print(f"[PATCH-STEP7] Patched: {TARGET}")
    print("[PATCH-STEP7] Done.")

if __name__ == "__main__":
    main()