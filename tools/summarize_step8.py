from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help=r"Run dir, e.g. results\step8_micro_v3")
    ap.add_argument("--arm", default="A3")
    args = ap.parse_args()

    run = Path(args.run)
    arm_dir = run / args.arm
    if not arm_dir.exists():
        raise SystemExit(f"[ERR] arm dir not found: {arm_dir}")

    files = sorted(arm_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"[ERR] no jsonl in: {arm_dir}")

    fn = files[-1]
    rows = 0
    did = 0
    cs = []
    sat = []
    lam = []
    nat_none = 0

    with fn.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows += 1
            if r.get("did_update") is True:
                did += 1
            if r.get("cos_spread") is not None:
                cs.append(float(r["cos_spread"]))
            if r.get("critic_saturation") is not None:
                sat.append(float(r["critic_saturation"]))
            if r.get("lambda_cit_eff") is not None:
                lam.append(float(r["lambda_cit_eff"]))
            if r.get("a_nat_none") is True:
                nat_none += 1

    def stats(xs):
        if not xs:
            return None
        return (min(xs), sum(xs) / len(xs), max(xs), len(set(round(x, 6) for x in xs)))

    print(f"file={fn}")
    print(f"rows={rows}")
    print(f"did_update={did}  rate={did / rows if rows else 0.0:.3f}")
    print(f"a_nat_none_count={nat_none}")
    print(f"lambda_cit_eff stats(min,mean,max,unique~)={stats(lam)}")
    print(f"cos_spread stats(min,mean,max,unique~)={stats(cs)}")
    print(f"critic_saturation stats(min,mean,max,unique~)={stats(sat)}")


if __name__ == "__main__":
    main()