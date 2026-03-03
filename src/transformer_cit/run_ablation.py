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
import copy
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
    critics_path: Optional[Path],
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
    # Step-8 CIT params (guardrails)
    cit_batch_size = int(ab_cfg.get("ablation", {}).get("cit_batch_size", 8))
    lambda_preserve = float(ab_cfg.get("ablation", {}).get("lambda_preserve", 1.0))
    sat_threshold = float(ab_cfg.get("ablation", {}).get("sat_threshold", 0.95))
    stop_cos_spread = float(ab_cfg.get("ablation", {}).get("stop_cos_spread", 0.99))
    stop_critic_saturation = float(ab_cfg.get("ablation", {}).get("stop_critic_saturation", 0.90))

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
    critics = CriticEnsemble(K=K, d=d).to(device)
    a_key = "a1"
    
    # Optional: load calibrated critics checkpoint (from train_critics.py)
    if critics_path is not None and critics_path.exists():
        ckp = torch.load(str(critics_path), map_location=device)
        sd = ckp.get("state_dict", None)
        meta = ckp.get("meta", {}) or {}
        rule_names = ckp.get("rule_names", None)
        if isinstance(rule_names, list) and len(rule_names) > 0:
            K = len(rule_names)
            critics = CriticEnsemble(K=K, d=d).to(device)
        if sd is None:
            raise RuntimeError(f"Critics checkpoint missing state_dict: {critics_path}")
        critics.load_state_dict(sd)
        a_key = str(meta.get("a_key", "a1"))
        if a_key not in ("a1", "a2", "a3"):
            a_key = "a1"
        print(f"[INFO] Loaded critics from: {critics_path} (a_key={a_key}, K={K}, d={d})")
    
    critics.freeze()
    critics.eval()
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
    # Step-8: reference (NAT) heads for preserve term (NO backbone clone)
    ref_heads_sd = None
    if hasattr(model, "probe_heads"):
        ref_heads_sd = [
            {k: v.detach().cpu().clone() for k, v in ph.state_dict().items()}
            for ph in model.probe_heads
        ]

    # Trainable heads on TRAIN model (backbone frozen)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "probe_heads"):
        for ph in model.probe_heads:
            for p in ph.parameters():
                p.requires_grad = True
    trainable_heads = [p for p in model.parameters() if p.requires_grad]
    opt_heads = (
        torch.optim.Adam(
            trainable_heads,
            lr=float(m_cfg.get("cit", {}).get("lr_heads", 1e-3)),
        )
        if trainable_heads
        else None
    )

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

            # Step-8 debug: effective lambda_cit
            lambda_cit_eff = float(
                weights.get(
                    "lambda_cit",
                    weights.get("lambda_cit_forge", getattr(sched.config, "lambda_cit_forge", 0.0)),
                )
            )
            # Tokenize + forward (B=cit_batch_size)
            prompts_b = [task.prompt] + [tasks[(step + j) % len(tasks)].prompt for j in range(1, cit_batch_size)]
            tok = model.tokenizer(
                prompts_b,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(m_cfg.get("cit", {}).get("max_length", 256)),
            )
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)  # dict a1/a2/a3
            a1 = out.get("a1", None)

            a_crit = out.get(a_key, None)
            a_nat = None
            if ref_heads_sd is not None and hasattr(model, "probe_heads"):
                # Save current heads
                cur_sd = [
                    {k: v.detach().cpu().clone() for k, v in ph.state_dict().items()}
                    for ph in model.probe_heads
                ]
                n = min(len(model.probe_heads), len(ref_heads_sd))
                # Load reference heads
                for i in range(n):
                    model.probe_heads[i].load_state_dict(ref_heads_sd[i])
                # Forward with reference heads (backbone is already no_grad inside CITModel)
                with torch.no_grad():
                    out_ref = model(input_ids=input_ids, attention_mask=attention_mask)
                a_nat = out_ref.get(a_key, None)
                # Restore current heads
                for i in range(n):
                    model.probe_heads[i].load_state_dict(cur_sd[i])

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
            if a_crit is not None:
                welfare = float(critics(a_crit)["aggregate"].mean().item())
                violation = 1.0 if welfare < tau else 0.0
                viol_hist.append(float(violation))
            # Losses (monitor-only; not training yet)
            loss_total = torch.tensor(0.0, device=device)
            loss_preserve = torch.tensor(0.0, device=device)
            cos_spread = None
            critic_saturation = None
            did_update = False

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


            if "L_welfare" in losses_enabled and a_crit is not None:
                # v0.1: target h_C = 1.0
                cw = critics(a_crit)["aggregate"].unsqueeze(-1)  # [B,1]
                hC = torch.ones_like(cw)
                loss_w = L_welfare(cw, hC) * weights.get("lambda_welfare", 0.0)
                loss_total = loss_total + loss_w
            else:
                loss_w = torch.tensor(0.0, device=device)
            if (
                "L_CIT" in losses_enabled
                and a_crit is not None
                and a_nat is not None
                and opt_heads is not None
                and lambda_cit_eff > 0.0
            ):
                cos_spread = _cos_spread(a_crit.detach())
                with torch.no_grad():
                    agg_now = critics(a_crit.detach())["aggregate"]
                    critic_saturation = float((agg_now > sat_threshold).float().mean().item())

                if (cos_spread is not None and cos_spread > stop_cos_spread) or (
                    critic_saturation is not None and critic_saturation > stop_critic_saturation
                ):
                    opt_heads = None
                    loss_c = torch.tensor(0.0, device=device)
                else:
                    loss_c = L_cit(a_crit, critics) * lambda_cit_eff
                    loss_preserve = lambda_preserve * (a_crit - a_nat.detach()).pow(2).sum(dim=-1).mean()
                    loss_update = loss_c + loss_preserve
                    opt_heads.zero_grad(set_to_none=True)
                    loss_update.backward()
                    opt_heads.step()
                    did_update = True

                loss_total = loss_total + loss_c + loss_preserve
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
                "loss_preserve": float(loss_preserve.item()),
                "cos_spread": cos_spread,
                "critic_saturation": critic_saturation,
                "did_update": did_update,
                "a_key_used": a_key,
                "lambda_cit_eff": lambda_cit_eff,
                "opt_heads_active": (opt_heads is not None),
                "a_nat_none": (a_nat is None),

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




def _cos_spread(x: torch.Tensor) -> float:
    """Mean pairwise cosine similarity in a batch. x: [B,d]."""
    if x is None:
        return float('nan')
    B = int(x.shape[0])
    if B < 2:
        return float('nan')
    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    sim = x @ x.t()
    return float(((sim.sum() - sim.diag().sum()) / (B * (B - 1))).item())
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default="configs/ablation_v0.yaml", help="Ablation YAML")
    ap.add_argument("--model", default="configs/gemma3_4b_cpu.yaml", help="Model YAML")
    ap.add_argument("--out", default="results/ablation_v0", help="Output directory")
    ap.add_argument("--arms", default="A0,A1,A2,A3", help="Comma-separated arms")
    ap.add_argument("--dry", action="store_true", help="Dry run (no HF model load)")
    ap.add_argument("--anchor", default="", help="Path to anchor .pt (mu_align) generated by anchor_offline")
    ap.add_argument("--heads", default="", help="Path to probe-head checkpoint (.pt) produced by train_heads.py")
    ap.add_argument("--critics", default="", help="Path to critics checkpoint (.pt) produced by train_critics.py")

    args = ap.parse_args()

    ab_cfg = load_yaml(Path(args.ablation))
    m_cfg = load_yaml(Path(args.model))
    anchor_path = Path(args.anchor) if args.anchor else None
    heads_path = Path(args.heads) if args.heads else None


    critics_path = Path(args.critics) if args.critics else None
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    for arm_id in arm_list_from_args(args.arms):
                run_arm(arm_id, ab_cfg, m_cfg, out_dir / arm_id, anchor_path=anchor_path, heads_path=heads_path, critics_path=critics_path, dry=args.dry)


    print(f"Done. Logs under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# PATCH_STEP8_V3_APPLIED

# PATCH_STEP8_MEMFIX_APPLIED

# PATCH_STEP8_DEBUG_LAMBDA_APPLIED
