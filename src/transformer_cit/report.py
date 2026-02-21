"""
CIT Ablation Report

Reads JSONL logs produced by src.transformer_cit.run_ablation and writes:
- summary_table.md (Markdown table, paper-ready)
- drift_curve_<ARM>.csv (optional, simple numeric export; no plotting deps)

Designed to be robust on Windows and to work with the current log keys:
  S_id01, S_id_anchor01, violation, welfare, tau, step, arm, timestamp, etc.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Returns (mean, std) with population std (ddof=0)."""
    if not values:
        return None, None
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return m, math.sqrt(var)


def _fmt_float(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{nd}f}"


def _fmt_mean_std(m: Optional[float], s: Optional[float], nd: int = 4) -> str:
    if m is None:
        return "N/A"
    if s is None:
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} +/- {s:.{nd}f}"


@dataclass
class ArmSummary:
    arm: str
    n_steps: int
    tau: Optional[float]
    s_id01_mean: Optional[float]
    s_id01_std: Optional[float]
    s_id_anchor_mean: Optional[float]
    s_id_anchor_std: Optional[float]
    violation_rate: Optional[float]
    violation_auc: Optional[float]
    welfare_final: Optional[float]
    log_path: Path


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _find_log_file(arm_dir: Path) -> Path:
    # Expect a*_log.jsonl but allow any *_log.jsonl
    candidates = sorted(arm_dir.glob("*_log.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No *_log.jsonl found under: {arm_dir}")
    if len(candidates) == 1:
        return candidates[0]
    # If multiple, pick the largest (most steps)
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def _summarize_arm(run_dir: Path, arm: str) -> ArmSummary:
    arm_dir = run_dir / arm
    log_path = _find_log_file(arm_dir)
    rows = _read_jsonl(log_path)

    # Series
    s_id01 = [float(r["S_id01"]) for r in rows if _is_num(r.get("S_id01"))]
    s_id_anchor = [
        float(r["S_id_anchor01"]) for r in rows if _is_num(r.get("S_id_anchor01"))
    ]
    violation = [float(r["violation"]) for r in rows if _is_num(r.get("violation"))]
    welfare = [float(r["welfare"]) for r in rows if _is_num(r.get("welfare"))]
    tau_vals = [float(r["tau"]) for r in rows if _is_num(r.get("tau"))]

    n_steps = len(rows)

    s_id01_mean, s_id01_std = _mean_std(s_id01)
    s_id_anchor_mean, s_id_anchor_std = _mean_std(s_id_anchor)

    violation_rate = (sum(violation) / len(violation)) if violation else None
    violation_auc = (sum(violation)) if violation else None  # dt=1 per step

    welfare_final = welfare[-1] if welfare else None

    tau = None
    if tau_vals:
        # choose the most common (robust if logged consistently)
        c = Counter(round(v, 6) for v in tau_vals)
        tau = float(c.most_common(1)[0][0])

    return ArmSummary(
        arm=arm,
        n_steps=n_steps,
        tau=tau,
        s_id01_mean=s_id01_mean,
        s_id01_std=s_id01_std,
        s_id_anchor_mean=s_id_anchor_mean,
        s_id_anchor_std=s_id_anchor_std,
        violation_rate=violation_rate,
        violation_auc=violation_auc,
        welfare_final=welfare_final,
        log_path=log_path,
    )


def _write_markdown_table(out_path: Path, summaries: List[ArmSummary]) -> None:
    lines: List[str] = []
    lines.append("# Ablation Summary\n")
    lines.append(
        "| arm | n_steps | tau | S_id01 (mean±std) | S_id_anchor01 (mean±std) | violation_rate | violation_auc | welfare_final | log |\n"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    for s in summaries:
        lines.append(
            "| {arm} | {n} | {tau} | {sid} | {sida} | {vr} | {va} | {wf} | {log} |\n".format(
                arm=s.arm,
                n=s.n_steps,
                tau=_fmt_float(s.tau, 3),
                sid=_fmt_mean_std(s.s_id01_mean, s.s_id01_std, 6),
                sida=_fmt_mean_std(s.s_id_anchor_mean, s.s_id_anchor_std, 6),
                vr=_fmt_float(s.violation_rate, 4),
                va=_fmt_float(s.violation_auc, 2),
                wf=_fmt_float(s.welfare_final, 4),
                log=s.log_path.as_posix(),
            )
        )

    out_path.write_text("".join(lines), encoding="utf-8")


def _write_drift_csv(run_dir: Path, arm: str, out_csv: Path) -> None:
    arm_dir = run_dir / arm
    log_path = _find_log_file(arm_dir)
    rows = _read_jsonl(log_path)

    cols = ["step", "S_id01", "S_id_anchor01", "welfare", "violation", "tau"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        required=True,
        help="Run directory under results/, e.g. results\\real_anchor_ablation_v0",
    )
    ap.add_argument(
        "--arms",
        default="",
        help="Comma-separated arms to include (default: all subfolders A0..A9 found)",
    )
    ap.add_argument(
        "--drift_arm",
        default="A3",
        help="Which arm to export drift CSV for (default: A3). Set empty to disable.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    if args.arms.strip():
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    else:
        # auto-detect arm folders like A0, A1, ...
        arms = sorted([p.name for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("A")])

    summaries: List[ArmSummary] = []
    for arm in arms:
        try:
            summaries.append(_summarize_arm(run_dir, arm))
        except Exception as e:
            # Keep report robust: include arm with N/A if it fails
            summaries.append(
                ArmSummary(
                    arm=arm,
                    n_steps=0,
                    tau=None,
                    s_id01_mean=None,
                    s_id01_std=None,
                    s_id_anchor_mean=None,
                    s_id_anchor_std=None,
                    violation_rate=None,
                    violation_auc=None,
                    welfare_final=None,
                    log_path=Path(f"{arm}/<missing>"),
                )
            )
            print(f"[WARN] Could not summarize {arm}: {e}")

    out_md = run_dir / "summary_table.md"
    _write_markdown_table(out_md, summaries)
    print(f"OK: wrote {out_md}")

    drift_arm = args.drift_arm.strip()
    if drift_arm:
        out_csv = run_dir / f"drift_curve_{drift_arm}.csv"
        try:
            _write_drift_csv(run_dir, drift_arm, out_csv)
            print(f"OK: wrote {out_csv}")
        except Exception as e:
            print(f"[WARN] Could not write drift CSV for {drift_arm}: {e}")


if __name__ == "__main__":
    main()
