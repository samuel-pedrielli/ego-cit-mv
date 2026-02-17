"""
run_ablation.py (v0)

Minimal runner for CIT transformer ablation arms A0-A3 as specified in spec.md.

- Loads YAML configs (ablation_v0.yaml + optional model config).
- Instantiates backbone wrapper + CIT heads (where enabled).
- Runs a small promptpack loop (stub) and logs JSONL per step.
- Produces a summary TXT with basic metrics.

This file is intentionally simple: it provides a working skeleton for the next iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Local modules
from .model import CITModelConfig, CITModel
from .critics import CriticEnsemble
from .losses import LossConfig, compute_losses
from .schedule import ForgeAnchorPreserveSchedule, ScheduleConfig


# -------------------------
# Utilities
# -------------------------

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------
# Promptpack (v0 stub)
# -------------------------

@dataclass
class PromptTask:
    task_id: str
    prompt: str


def load_promptpack(promptpack_path: Optional[Path]) -> List[PromptTask]:
    """
    v0: if promptpack_path is provided and exists, it must be JSONL:
      {"task_id": "...", "prompt": "..."}
    If absent, we create a tiny built-in pack.
    """
    if promptpack_path is None or not promptpack_path.exists():
        return [
            PromptTask(task_id="toy_01", prompt="You are a helpful assistant. Summarize the importance of safety."),
            PromptTask(task_id="toy_02", prompt="Explain why internal monitoring might detect drift earlier than outputs."),
        ]

    tasks: List[PromptTask] = []
    with promptpack_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks.append(PromptTask(task_id=str(obj["task_id"]), prompt=str(obj["prompt"])))
    return tasks


# -------------------------
# Arm config
# -------------------------

@dataclass
class Arm:
    arm_id: str  # A0/A1/A2/A3
    enable_identity: bool
    enable_welfare: bool
    enable_cit: bool


def arm_from_id(arm_id: str) -> Arm:
    arm_id = arm_id.upper().strip()
    if arm_id == "A0":
        return Arm("A0", enable_identity=False, enable_welfare=False, enable_cit=False)
    if arm_id == "A1":
        return Arm("A1", enable_identity=True, enable_welfare=False, enable_cit=False)
    if arm_id == "A2":
        return Arm("A2", enable_identity=True, enable_welfare=True, enable_cit=False)
    if arm_id == "A3":
        return Arm("A3", enable_identity=True, enable_welfare=True, enable_cit=True)
    raise ValueError(f"Unknown arm_id: {arm_id}")


# -------------------------
# Main run loop (skeleton)
# -------------------------

def run_arm(
    arm: Arm,
    cfg: Dict[str, Any],
    out_dir: Path,
) -> Path:
    """
    Runs a single arm and returns the path to the JSONL log.
    """
    ensure_dir(out_dir)
    log_path = out_dir / f"{arm.arm_id.lower()}_log.jsonl"

    # ---- Parse configs (minimal)
    model_cfg = CITModelConfig.from_dict(cfg["model"])
    loss_cfg = LossConfig.from_dict(cfg.get("losses", {}))
    sched_cfg = ScheduleConfig.from_dict(cfg.get("schedule", {}))

    # Arm overrides
    model_cfg.enable_identity = arm.enable_identity
    loss_cfg.enable_welfare = arm.enable_welfare
    loss_cfg.enable_cit = arm.enable_cit

    # ---- Instantiate model wrapper
    model = CITModel(model_cfg)

    # Critics (frozen)
    critics = CriticEnsemble.from_dict(cfg.get("critics", {}))
    critics.freeze()

    # Schedule object (controls phase)
    schedule = ForgeAnchorPreserveSchedule(sched_cfg)

    # Promptpack
    promptpack_path = Path(cfg["promptpack"]) if "promptpack" in cfg else None
    tasks = load_promptpack(promptpack_path)

    # Run
    step = 0
    s_id_values: List[float] = []
    viol_values: List[float] = []

    for task in tasks:
        # v0: a "rollout" is just N steps repeating the prompt.
        rollout_steps = int(cfg.get("rollout_steps", 8))

        for t in range(rollout_steps):
            step += 1
            phase = schedule.phase(step)

            # ---- Forward
            # v0: We do not generate tokens yet; we only run a forward pass on the prompt.
            # CITModel.forward returns:
            #  - a1: identity vector [B,d] or None
            #  - s_id01: float in [0,1] or None
            #  - aux: dict with tensors
            out = model.forward_text(task.prompt)

            s_id01 = out.get("s_id01")
            if s_id01 is not None:
                s_id_values.append(float(s_id01))

            # ---- Welfare / violation proxy (stub)
            # v0: no real constitutional evaluator; if enabled, we use critics on a1.
            violation = None
            if arm.enable_welfare and out.get("a1") is not None:
                w_t = critics.welfare(out["a1"], context={"prompt": task.prompt})
                tau = float(cfg.get("tau", 0.9))
                violation = 1.0 if float(w_t) < tau else 0.0
                viol_values.append(float(violation))

            # ---- Loss computation (only meaningful when training is implemented)
            # v0: compute_losses returns scalars but we do not yet run optimizer steps.
            loss_dict = compute_losses(
                out=out,
                loss_cfg=loss_cfg,
                critics=critics,
                phase=phase,
                context={"prompt": task.prompt},
            )

            row = {
                "timestamp": now_ts(),
                "arm": arm.arm_id,
                "task_id": task.task_id,
                "step": step,
                "t_in_task": t,
                "phase": phase,
                "s_id01": float(s_id01) if s_id01 is not None else None,
                "violation": float(violation) if violation is not None else None,
                "loss_total": float(loss_dict.get("loss_total", 0.0)),
                "loss_id": float(loss_dict.get("loss_id", 0.0)),
                "loss_welfare": float(loss_dict.get("loss_welfare", 0.0)),
                "loss_cit": float(loss_dict.get("loss_cit", 0.0)),
            }
            write_jsonl(log_path, row)

    # Write summary
    summary_path = out_dir / f"{arm.arm_id.lower()}_summary.txt"
    mean_s = sum(s_id_values) / len(s_id_values) if s_id_values else float("nan")
    mean_v = sum(viol_values) / len(viol_values) if viol_values else float("nan")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"arm: {arm.arm_id}\n")
        f.write(f"timestamp: {now_ts()}\n")
        f.write(f"steps: {step}\n")
        f.write(f"mean_S_id01: {mean_s}\n")
        f.write(f"mean_violation: {mean_v}\n")

    return log_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ablation_v0.yaml", help="Path to ablation YAML")
    ap.add_argument("--out", default="results/ablation_v0", help="Output directory")
    ap.add_argument("--arms", default="A0,A1,A2,A3", help="Comma-separated arms")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    arms = [arm_from_id(x) for x in args.arms.split(",") if x.strip()]
    for arm in arms:
        arm_out = out_dir / arm.arm_id
        run_arm(arm, cfg, arm_out)

    print(f"Done. Logs under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
