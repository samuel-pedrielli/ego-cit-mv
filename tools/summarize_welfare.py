from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory (e.g., results\\critics_wiring_check_a3)")
    args = ap.parse_args()

    root = Path(args.run)
    if not root.exists():
        raise SystemExit(f"[ERR] run dir not found: {root}")

    files = list(root.rglob("*.jsonl"))
    print(f"run={root}")
    print(f"jsonl_files={len(files)}")

    welfare_by = defaultdict(list)
    viol_by = defaultdict(list)
    null_welfare = defaultdict(int)
    total = 0

    # Also track unique welfare per arm (rounded) to confirm prompt sensitivity
    uniq = defaultdict(set)

    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                total += 1
                arm = r.get("arm", "?")
                w = r.get("welfare", None)
                v = r.get("violation", None)
                if w is None:
                    null_welfare[arm] += 1
                else:
                    wf = float(w)
                    welfare_by[arm].append(wf)
                    uniq[arm].add(round(wf, 6))
                if v is not None:
                    try:
                        viol_by[arm].append(float(v))
                    except Exception:
                        pass

    print(f"rows={total}")
    arms = sorted(set(list(welfare_by.keys()) + list(null_welfare.keys()) + list(viol_by.keys())))

    for arm in arms:
        ws = welfare_by.get(arm, [])
        vs = viol_by.get(arm, [])
        nw = null_welfare.get(arm, 0)
        if ws:
            print(
                f"{arm}: welfare n={len(ws)} min={min(ws):.4f} mean={statistics.fmean(ws):.4f} "
                f"max={max(ws):.4f} unique~={len(uniq[arm])} nulls={nw}"
            )
        else:
            print(f"{arm}: welfare n=0 nulls={nw}")
        if vs:
            print(f"    violation_rate={statistics.fmean(vs):.4f}  (n={len(vs)})")

    # Gate hint
    if "A3" in welfare_by and len(uniq["A3"]) > 1:
        print("[OK] Gate: welfare varies across prompts (A3 unique welfare > 1).")
    elif "A3" in welfare_by:
        print("[WARN] Gate: welfare did not vary (A3 unique welfare == 1). Investigate.")
    else:
        print("[WARN] Gate: no A3 welfare recorded (check enable_probes / a_key availability).")


if __name__ == "__main__":
    main()