"""
run_ablation.py (v0.1)

Runner skeleton aligned to current repo modules:
- model.py: CITModel
- critics.py: CriticEnsemble
- losses.py: IdentityLoss, IdentityStabilityLoss, WelfareLoss, CITLoss
- schedule.py: FAPConfig, ForgeAnchorPreserve

It supports:
- ablation config: configs/ablation_v0.yaml
- model config: configs/gemma3_4b_cpu.yaml
- logging JSONL + summary TXT

NOTE: This is still "monitor + plumbing" (no optimizer/training yet).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import yaml

from .model import CITModel
from .critics import CriticEnsemble
from .losses import IdentityLoss, IdentityStabilityLoss, WelfareLoss, CITLoss
from .schedule import FAPConfig, ForgeAnchorPreserve
from .anchor import AnchorStore



# -------------------------
# Utilities
# -------------------------

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cosine01(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity mapped from [-1,1] to [0,1]."""
    c = F.cosine_similarity(a, b, dim=-1)
    return (c + 1.0) / 2.0


# -------------------------
# Promptpack (v0)
# -------------------------

@dataclass
class PromptTask:
    task_id: str
    prompt: str


def load_promptpack(promptpack_path: Optional[Path]) -> List[PromptTask]:
    """
    If file exists, must be JSONL rows:
      {"task_id": "...", "prompt": "..."}
    Otherwise, tiny built-in pack.
    """
    if promptpack_path is None or not promptpack_path.exists():
        return [
            PromptTask("toy_01", "You are a helpful assistant. Summarize the importance of safety."),
            PromptTask("toy_02", "Explain why internal monitoring might detect drift earlier than outputs."),
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
# Arms
# -------------------------

def arm_list_from_args(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


# -------------------------
# Run
# -------------------------

def run_arm(
    arm_id: str,
    ab_cfg: Dict[str, Any],
    m_cfg: Dict[str, Any],
    out_dir: Path,
    anchor_path: Optional[Path],
    heads_path: Optional[Path],
    dry: bool,
) -> None:

    ensure_dir(out_dir)
    log_path = out_dir / f"{arm_id.lower()}_log.jsonl"
    summary_path = out_dir / f"{arm_id.lower()}_summary.txt"

    arm_cfg = ab_cfg["ablation"]["arms"][arm_id]
    enable_probes = bool(arm_cfg.get("enable_probes", True))
    losses_enabled = set(arm_cfg.get("losses", []))

    # Model config
    model_name = m_cfg["model"]["name"]
    tap_layers = list(m_cfg["model"]["tap_layers"]) if enable_probes else []
    d = int(m_cfg["model"].get("identity_dim", 64))
    pooling = str(m_cfg["model"].get("pooling", "mean"))
    use_mlp_heads = bool(m_cfg["model"].get("use_mlp_heads", False))

    device = str(m_cfg.get("training", {}).get("device", "cpu"))
    tau = float(ab_cfg.get("ablation", {}).get("tau_welfare", 0.9))  # violation threshold (configurable)

    tau_crit = float(m_cfg.get("cit", {}).get("tau_crit", 0.7))
    eps_cit = float(m_cfg.get("cit", {}).get("epsilon", 0.01))

    # Schedule config (Forge/Preserve only; Anchor is offline)
    fap_cfg = FAPConfig(
        forge_steps=int(m_cfg.get("schedule", {}).get("forge_steps", 50)),
        preserve_steps=int(m_cfg.get("schedule", {}).get("preserve_steps", 20)),
        lambda_cit_forge=float(m_cfg.get("schedule", {}).get("lambda_cit_forge", 0.2)),
        lambda_cit_decay_end=float(m_cfg.get("schedule", {}).get("lambda_cit_decay_end", 0.01)),
        lambda_self_ramp_end=float(m_cfg.get("schedule", {}).get("lambda_self_ramp_end", 0.2)),
        s_id_floor=float(m_cfg.get("schedule", {}).get("s_id_floor", 0.9)),
        early_stop_windows=int(m_cfg.get("schedule", {}).get("early_stop_windows", 2)),
    )
    sched = ForgeAnchorPreserve(fap_cfg)

    # Critics
    K = int(m_cfg.get("critics", {}).get("num_rules", 5))
    critics = CriticEnsemble(K=K, d=d)
    critics.freeze()
    # Optional anchor (mu_align) for L_self and S_id logging
    mu_anchor = None
    if anchor_path is not None and anchor_path.exists():
        store = AnchorStore.load_from_file(str(anchor_path))
        mu_anchor = store.mu_align.to(device)  # shape [d]

    # Loss modules (weights are handled by sched.get_loss_weights())
    L_id = IdentityLoss(lambda_c=1.0)
    L_self = IdentityStabilityLoss(mu_c=1.0)
    L_welfare = WelfareLoss()
    L_cit = CITLoss(tau_crit=tau_crit, epsilon=eps_cit)

    # Promptpack
    promptpack_path = Path(ab_cfg["ablation"].get("promptpack", "")) if ab_cfg["ablation"].get("promptpack") else None
    tasks = load_promptpack(promptpack_path)
    rollout_steps = int(ab_cfg.get("rollout_steps", 6))

    # Dry mode: no HF model load, just log structure
    if dry:
        step = 0
        for task in tasks:
            for t in range(rollout_steps):
                step += 1
                row = {
                    "timestamp": now_ts(),
                    "arm": arm_id,
                    "task_id": task.task_id,
                    "step": step,
                    "t_in_task": t,
                    "phase": sched.phase, 
                    "tau": tau,
                    "S_id01": None,
                    "S_id_anchor01": None,
                    "welfare": None,
                    "violation": None,
                    "loss_total": 0.0,
                }
                write_jsonl(log_path, row)
        with summary_path.open("w", encoding="utf-8") as f:
            f.write(f"arm: {arm_id}\nsteps: {step}\nDRY_RUN: true\n")
        return

    # Real mode: load backbone (may download)
    model = CITModel(
        model_name=model_name,
        tap_layers=tap_layers,
        d=d,
        pooling=pooling,
        use_mlp_heads=use_mlp_heads,
    ).to(device)

    # Optional: load trained probe heads (from train_heads.py)
    if heads_path is not None and heads_path.exists():
        ckpt = torch.load(str(heads_path), map_location=device)
        heads_sd = ckpt.get("probe_heads", [])
        if hasattr(model, "probe_heads") and heads_sd:
            n = min(len(model.probe_heads), len(heads_sd))
            for i in range(n):
                model.probe_heads[i].load_state_dict(heads_sd[i])
            print(f"[INFO] Loaded {n} probe head(s) from: {heads_path}")
        else:
            print(f"[WARN] No probe_heads found in checkpoint or model: {heads_path}")

    prev_a: Dict[str, torch.Tensor] = {}
    s_id_hist: List[float] = []
    viol_hist: List[float] = []
    step = 0

    for task in tasks:
        for t in range(rollout_steps):
            step += 1

            # Phase transition
            if sched.phase == "forge" and step >= sched.config.forge_steps:
                sched.transition_to_preserve()
            weights = sched.get_loss_weights()

            # Tokenize + forward (B=1)
            tok = model.tokenizer(task.prompt, return_tensors="pt")
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)  # dict a1/a2/a3
            a1 = out.get("a1", None)

            # S_id (a1_t vs a1_{t-1})
            S_id01 = None
            if a1 is not None and "a1" in prev_a:
                S_id01 = float(cosine01(a1, prev_a["a1"]).mean().item())
                s_id_hist.append(S_id01)

            # S_id w.r.t. offline anchor (if provided)
            S_id_anchor01 = None
            if a1 is not None and mu_anchor is not None:
                mu_b = mu_anchor.unsqueeze(0).expand_as(a1)
                S_id_anchor01 = float(cosine01(a1, mu_b).mean().item())

            # Welfare proxy: use critic aggregate as w_t in [0,1]
            welfare = None
            violation = None
            if a1 is not None:
                welfare = float(critics(a1)["aggregate"].mean().item())
                violation = 1.0 if welfare < tau else 0.0
                viol_hist.append(float(violation))

            # Losses (monitor-only; not training yet)
            loss_total = torch.tensor(0.0, device=device)

            if "L_id" in losses_enabled and prev_a and out:
                loss_id = L_id(out, prev_a) * weights.get("lambda_id", 0.0)
                loss_total = loss_total + loss_id
            else:
                loss_id = torch.tensor(0.0, device=device)

            if "L_self" in losses_enabled and a1 is not None:
                # Prefer offline anchor if provided; fallback keeps runner usable without anchor
                if mu_anchor is not None:
                    mu_target = mu_anchor.unsqueeze(0).expand_as(a1)
                else:
                    mu_target = a1.detach()
                loss_self = L_self(a1, mu_target) * weights.get("lambda_self", 0.0)
                loss_total = loss_total + loss_self
            else:
                loss_self = torch.tensor(0.0, device=device)


            if "L_welfare" in losses_enabled and a1 is not None:
                # v0.1: target h_C = 1.0
                cw = critics(a1)["aggregate"].unsqueeze(-1)  # [B,1]
                hC = torch.ones_like(cw)
                loss_w = L_welfare(cw, hC) * weights.get("lambda_welfare", 0.0)
                loss_total = loss_total + loss_w
            else:
                loss_w = torch.tensor(0.0, device=device)

            if "L_CIT" in losses_enabled and a1 is not None:
                loss_c = L_cit(a1, critics) * weights.get("lambda_cit", 0.0)
                loss_total = loss_total + loss_c
            else:
                loss_c = torch.tensor(0.0, device=device)

            row = {
                "timestamp": now_ts(),
                "arm": arm_id,
                "task_id": task.task_id,
                "step": step,
                "t_in_task": t,
                "phase": sched.phase,
                "tau": tau,
                "S_id01": S_id01,
                "S_id_anchor01": S_id_anchor01,
                "welfare": welfare,
                "violation": violation,
                "loss_total": float(loss_total.item()),
                "loss_id": float(loss_id.item()),
                "loss_self": float(loss_self.item()),
                "loss_welfare": float(loss_w.item()),
                "loss_cit": float(loss_c.item()),
            }
            write_jsonl(log_path, row)

            # Update prev
            prev_a = {k: v.detach() for k, v in out.items()}

            # Early stop (monitor)
            if sched.should_early_stop(s_id_hist):
                break

        sched.advance()

    mean_s = sum(s_id_hist) / len(s_id_hist) if s_id_hist else float("nan")
    mean_v = sum(viol_hist) / len(viol_hist) if viol_hist else float("nan")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"arm: {arm_id}\n")
        f.write(f"timestamp: {now_ts()}\n")
        f.write(f"steps: {step}\n")
        f.write(f"mean_S_id01: {mean_s}\n")
        f.write(f"mean_violation: {mean_v}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default="configs/ablation_v0.yaml", help="Ablation YAML")
    ap.add_argument("--model", default="configs/gemma3_4b_cpu.yaml", help="Model YAML")
    ap.add_argument("--out", default="results/ablation_v0", help="Output directory")
    ap.add_argument("--arms", default="A0,A1,A2,A3", help="Comma-separated arms")
    ap.add_argument("--dry", action="store_true", help="Dry run (no HF model load)")
    ap.add_argument("--anchor", default="", help="Path to anchor .pt (mu_align) generated by anchor_offline")
    ap.add_argument("--heads", default="", help="Path to probe-head checkpoint (.pt) produced by train_heads.py")

    args = ap.parse_args()

    ab_cfg = load_yaml(Path(args.ablation))
    m_cfg = load_yaml(Path(args.model))
    anchor_path = Path(args.anchor) if args.anchor else None
    heads_path = Path(args.heads) if args.heads else None


    out_dir = Path(args.out)
    ensure_dir(out_dir)

    for arm_id in arm_list_from_args(args.arms):
                run_arm(arm_id, ab_cfg, m_cfg, out_dir / arm_id, anchor_path=anchor_path, heads_path=heads_path, dry=args.dry)


    print(f"Done. Logs under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
