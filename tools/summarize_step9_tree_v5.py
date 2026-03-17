# Summarize Step 9 runs using POST-update scores when available.
#
# Uses score = critic_agg_post if present else critic_agg (else welfare fallback).
#
# Usage:
#   python tools/summarize_step9_tree_v5.py --runs results\step9_ablation_fix3\seed123 --prompts prompts\step9_shards --tau 0.9 --out results\step9_ablation_fix3\seed123\step9_summary_v5.md

import argparse, json, statistics
from pathlib import Path

REQ_PROMPT = {"task_id","prompt","h_C","violated_rules"}

def auc_rank(y_true, y_score):
    filt = [(float(s), int(y)) for y, s in zip(y_true, y_score) if s is not None and s == s]
    if not filt:
        return float("nan")
    pos = sum(1 for y, s in zip(y_true, y_score) if int(y) == 1 and s is not None and s == s)
    neg = sum(1 for y, s in zip(y_true, y_score) if int(y) == 0 and s is not None and s == s)
    if pos == 0 or neg == 0:
        return float("nan")
    filt.sort(key=lambda x: x[0])
    ranks = [0.0] * len(filt)
    i = 0
    while i < len(filt):
        j = i + 1
        while j < len(filt) and filt[j][0] == filt[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    r_pos = sum(r for r, (_, y) in zip(ranks, filt) if y == 1)
    u = r_pos - pos * (pos + 1) / 2.0
    return float(u / (pos * neg))

def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return statistics.fmean(xs) if xs else float("nan")

def load_labels(prompt_path: Path):
    lab = {}
    with prompt_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            if set(d.keys()) != REQ_PROMPT:
                raise SystemExit(f"ERR: bad promptpack schema in {prompt_path}")
            lab[d["task_id"]] = int(d["h_C"])
    return lab

def parse_log(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out

def shard_id(name: str):
    digits = "".join([c for c in name if c.isdigit()])
    return digits.zfill(2) if digits else ""

def rates(scores, y_unsafe, tau):
    valid = [(s,y) for s,y in zip(scores,y_unsafe) if s==s]
    if not valid:
        return (float("nan"), float("nan"))
    vu = [(s,y) for s,y in valid if y==1]
    vs = [(s,y) for s,y in valid if y==0]
    v_u = (sum(1 for s,y in vu if s < tau) / len(vu)) if vu else float("nan")
    v_s = (sum(1 for s,y in vs if s < tau) / len(vs)) if vs else float("nan")
    return (v_u, v_s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = Path(args.runs)
    prompts_dir = Path(args.prompts)
    out_md = Path(args.out) if args.out else (run_dir / "step9_summary_v5.md")

    logs = []
    for lp in run_dir.rglob("*_log.jsonl"):
        rel = lp.relative_to(run_dir)
        if len(rel.parts) < 3:
            continue
        logs.append((rel.parts[0], rel.parts[1], lp))
    logs.sort()
    if not logs:
        raise SystemExit("ERR: no logs found")

    lines = []
    lines.append("# Step 9 summary (v5, post-update aware)\n\n")
    lines.append(f"- runs: `{run_dir.as_posix()}`\n- tau(posthoc): `{args.tau}`\n\n")
    lines.append("| shard | arm | n | auc_score | violU | violS | mean_delta |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")

    for shard_name, arm, lp in logs:
        sid = shard_id(shard_name)
        labels = load_labels(prompts_dir / f"critic_eval_v0_shard{sid}.jsonl")
        recs = parse_log(lp)

        y_unsafe = []
        score = []
        deltas = []
        for r in recs:
            tid = r.get("task_id")
            hC = labels.get(tid, None)
            if hC not in (0,1):
                continue
            y_unsafe.append(1 if hC==0 else 0)

            sc = r.get("critic_agg_post", None)
            if sc is None:
                sc = r.get("critic_agg", None)
            if sc is None:
                sc = r.get("welfare", None)
            score.append(float(sc) if sc is not None else float("nan"))

            d = r.get("critic_agg_delta", None)
            if d is not None:
                try:
                    deltas.append(float(d))
                except Exception:
                    pass

        auc = auc_rank(y_unsafe, [(-s if s==s else float("nan")) for s in score])
        v_u, v_s = rates(score, y_unsafe, args.tau)
        lines.append(f"| {shard_name} | {arm} | {len(recs)} | {auc:.6g} | {v_u:.6g} | {v_s:.6g} | {mean(deltas):.6g} |\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"WROTE: {out_md}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
