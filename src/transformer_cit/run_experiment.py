"""
run_experiment.py — CIT ablation runner (v2, clean rewrite)
============================================================

Replaces run_ablation.py.  Written from scratch to fix:

1. Gradient scaling: LayerNorm in model.py makes probe head gradients
   independent of backbone hidden_size.  No lr tricks needed.
2. Preserve loss: .mean(dim=-1) not .sum(dim=-1).
3. Schedule: advance() called per training step; forge → preserve
   transition happens within the actual training budget.
4. Early stop: flag-based break from both loops.
5. Efficient preserve: re-uses pooled hidden states instead of
   running backbone twice per step.
6. A3R arm: random (uncalibrated) critics for sanity check.

Usage (standalone):
  python -m src.transformer_cit.run_experiment \\
    --ablation configs/step9_shards/ablation_step9_shard01_v2.yaml \\
    --model configs/gemma3_4b_gpu.yaml \\
    --out results/v2/seed123/shard01 \\
    --arms A0,A3 --seed 123

Usage (from pipeline_gpu.py):
  run_arm(..., preloaded_model=model, preloaded_device="cuda")
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .model import CITModel
from .critics import CriticEnsemble
from .losses import CITLoss
from .schedule import FAPConfig, ForgeAnchorPreserve
from .anchor import AnchorStore


# ── Utilities ─────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _cos01(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity in [0, 1]."""
    c = F.cosine_similarity(a, b, dim=-1)
    return float(((c + 1.0) / 2.0).mean().item())


def _spread(x: torch.Tensor) -> float:
    """Mean pairwise cosine similarity (batch diversity)."""
    if x is None or x.shape[0] < 2:
        return float("nan")
    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    sim = x @ x.t()
    B = x.shape[0]
    return float(((sim.sum() - sim.diag().sum()) / (B * (B - 1))).item())


# ── Promptpack ────────────────────────────────────────────

@dataclass
class PromptTask:
    task_id: str
    prompt: str


def load_promptpack(path: Optional[Path]) -> List[PromptTask]:
    if path is None or not path.exists():
        return [
            PromptTask("toy_01", "You are a helpful assistant. Summarize the importance of safety."),
            PromptTask("toy_02", "Explain why internal monitoring might detect drift earlier than outputs."),
        ]
    tasks: List[PromptTask] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks.append(PromptTask(str(obj["task_id"]), str(obj["prompt"])))
    return tasks


# ── Core runner ───────────────────────────────────────────

def run_arm(
    arm_id: str,
    ab_cfg: Dict[str, Any],
    m_cfg: Dict[str, Any],
    out_dir: Path,
    anchor_path: Optional[Path] = None,
    heads_path: Optional[Path] = None,
    critics_path: Optional[Path] = None,
    seed: int = 123,
    # Pipeline mode: pass pre-loaded model to avoid reloading
    preloaded_model: Optional[CITModel] = None,
    preloaded_device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single ablation arm (A0, A3, or A3R).

    Returns a dict with summary statistics.

    When preloaded_model is given, probe heads are saved at the start
    and restored at the end, so the model can be reused across arms.
    """

    _dir(out_dir)
    log_path = out_dir / f"{arm_id.lower()}_log.jsonl"
    posthoc_path = out_dir / f"{arm_id.lower()}_posthoc.jsonl"
    summary_path = out_dir / f"{arm_id.lower()}_posthoc_summary.txt"

    # ── Arm config ──
    arm_cfg = ab_cfg["ablation"]["arms"].get(arm_id, {})
    losses_enabled = set(arm_cfg.get("losses", []))
    use_random_critics = bool(arm_cfg.get("random_critics", False))
    # Convention: arm_id ending with "R" also means random critics
    if arm_id.upper().endswith("R"):
        use_random_critics = True

    # ── Model config ──
    model_name = m_cfg["model"]["name"]
    tap_layers = list(m_cfg["model"]["tap_layers"])
    d = int(m_cfg["model"].get("identity_dim", 64))
    pooling = str(m_cfg["model"].get("pooling", "mean"))
    use_mlp = bool(m_cfg["model"].get("use_mlp_heads", False))
    use_ln = bool(m_cfg["model"].get("use_layernorm", True))

    # ── Training params ──
    device = preloaded_device or str(m_cfg.get("training", {}).get("device", "cpu"))
    lambda_preserve = float(ab_cfg["ablation"].get("lambda_preserve", 0.3))
    max_grad_norm = float(ab_cfg["ablation"].get("max_grad_norm", 1.0))
    sat_threshold = float(ab_cfg["ablation"].get("sat_threshold", 0.95))
    stop_crit_sat = float(ab_cfg["ablation"].get("stop_critic_saturation", 1.0))
    tau_crit = float(m_cfg.get("cit", {}).get("tau_crit", 0.9))
    lr_heads = float(m_cfg.get("cit", {}).get("lr_heads", 1e-4))
    max_length = int(m_cfg.get("cit", {}).get("max_length", 256))
    cit_batch_size = int(ab_cfg["ablation"].get("cit_batch_size", 2))

    # ── Promptpack ──
    pp_str = ab_cfg["ablation"].get("promptpack", "")
    pp_path = Path(pp_str) if pp_str else None
    tasks = load_promptpack(pp_path)
    rollout_steps = int(ab_cfg.get("rollout_steps", 3))
    total_steps = rollout_steps * len(tasks)

    # ── Schedule (calibrated to actual step count) ──
    raw_forge = int(m_cfg.get("schedule", {}).get("forge_steps", 50))
    raw_preserve = int(m_cfg.get("schedule", {}).get("preserve_steps", 25))
    # If config values exceed total steps, auto-calibrate
    if raw_forge + raw_preserve > total_steps:
        forge_steps = int(total_steps * 0.67)
        preserve_steps = total_steps - forge_steps
        print(f"[INFO] Schedule auto-calibrated: forge={forge_steps}, "
              f"preserve={preserve_steps} (total={total_steps})")
    else:
        forge_steps = raw_forge
        preserve_steps = raw_preserve

    fap_cfg = FAPConfig(
        forge_steps=forge_steps,
        preserve_steps=preserve_steps,
        lambda_cit_forge=float(m_cfg.get("schedule", {}).get("lambda_cit_forge", 0.2)),
        lambda_cit_decay_end=float(m_cfg.get("schedule", {}).get("lambda_cit_decay_end", 0.01)),
        lambda_self_ramp_end=float(m_cfg.get("schedule", {}).get("lambda_self_ramp_end", 0.2)),
        s_id_floor=float(m_cfg.get("schedule", {}).get("s_id_floor", 0.9)),
        early_stop_windows=int(m_cfg.get("schedule", {}).get("early_stop_windows", 5)),
    )
    sched = ForgeAnchorPreserve(fap_cfg)

    # ── Critics ──
    K = int(m_cfg.get("critics", {}).get("num_rules", 5))
    a_key = "a1"  # default, overridden if checkpoint has meta
    critics = CriticEnsemble(K=K, d=d).to(device)

    if use_random_critics:
        # A3R sanity check: keep random weights, don't load checkpoint
        print(f"[INFO] {arm_id}: using RANDOM (uncalibrated) critics")
    elif critics_path is not None and critics_path.exists():
        ckp = torch.load(str(critics_path), map_location=device, weights_only=False)
        sd = ckp.get("state_dict")
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
        print(f"[INFO] Critics loaded: {critics_path} (a_key={a_key}, K={K})")

    critics.freeze()
    critics.eval()

    # ── Anchor (mu_align) for S_id logging ──
    mu_anchor = None
    if anchor_path is not None and anchor_path.exists():
        store = AnchorStore.load_from_file(str(anchor_path))
        mu_anchor = store.mu_align.to(device)

    # ── CIT loss ──
    L_cit = CITLoss(tau_crit=tau_crit)

    # ── Load or create model ──
    torch.manual_seed(seed)
    own_model = False
    if preloaded_model is not None:
        model = preloaded_model
    else:
        own_model = True
        q4 = bool(m_cfg["model"].get("quantize_4bit", False))
        dtype = str(m_cfg["model"].get("torch_dtype", "float32"))
        model = CITModel(
            model_name=model_name,
            tap_layers=tap_layers,
            d=d,
            pooling=pooling,
            use_mlp_heads=use_mlp,
            use_layernorm=use_ln,
            quantize_4bit=q4,
            torch_dtype=dtype,
        ).to(device)

    # ── Load trained probe heads ──
    if heads_path is not None and heads_path.exists():
        ckpt = torch.load(str(heads_path), map_location=device, weights_only=False)
        heads_sd = ckpt.get("probe_heads", [])
        if hasattr(model, "probe_heads") and heads_sd:
            n = min(len(model.probe_heads), len(heads_sd))
            for i in range(n):
                model.probe_heads[i].load_state_dict(heads_sd[i])
            print(f"[INFO] Loaded {n} probe head(s) from: {heads_path}")

    # ── Save initial head state (reference for preserve + post-hoc) ──
    initial_heads_sd = [
        {k: v.detach().cpu().clone() for k, v in ph.state_dict().items()}
        for ph in model.probe_heads
    ]

    # ── Reference probe heads for preserve loss ──
    # Separate frozen copies of the initial heads.  Applied to pooled
    # hidden states during training (no backbone re-run needed).
    import copy
    ref_probe_heads = copy.deepcopy(model.probe_heads)
    for p in ref_probe_heads.parameters():
        p.requires_grad = False
    ref_probe_heads = ref_probe_heads.to(device)
    ref_probe_heads.eval()

    # Index of the critic-targeted probe head
    crit_head_idx = int(a_key[1]) - 1  # "a3" → 2

    # ── Freeze backbone, enable probe head gradients ──
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "probe_heads"):
        for ph in model.probe_heads:
            for p in ph.parameters():
                p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt_heads = (
        torch.optim.Adam(trainable_params, lr=lr_heads)
        if trainable_params and "L_CIT" in losses_enabled
        else None
    )

    # ── Training loop (epoch-based, schedule per step) ──
    print(f"\n[INFO] {arm_id}: {len(tasks)} prompts × {rollout_steps} epochs "
          f"= {total_steps} steps | losses={losses_enabled}")

    s_id_hist: List[float] = []
    step = 0
    early_stopped = False
    prev_a: Dict[str, torch.Tensor] = {}

    for epoch in range(rollout_steps):
        if early_stopped:
            break

        for task in tasks:
            if early_stopped:
                break

            step += 1
            sched.advance()
            weights = sched.get_loss_weights()
            lambda_cit_eff = weights.get("lambda_cit", 0.0)

            # ── Tokenize (mini-batch: current task + neighbors) ──
            prompts_b = [task.prompt]
            for j in range(1, cit_batch_size):
                prompts_b.append(tasks[(step + j) % len(tasks)].prompt)

            tok = model.tokenizer(
                prompts_b, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            input_ids = tok["input_ids"].to(device)
            attn_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(device)

            # ── Forward (with pooled hidden states for preserve) ──
            out, pooled = model(input_ids, attn_mask, return_pooled=True)
            a_crit = out.get(a_key)
            a1 = out.get("a1")

            # ── Compute a_nat from reference heads (no backbone re-run) ──
            a_nat = None
            with torch.no_grad():
                h_pooled = pooled.get(f"h{crit_head_idx + 1}")
                if h_pooled is not None:
                    a_nat = ref_probe_heads[crit_head_idx](h_pooled.detach())

            # ── Metrics ──
            S_id01 = None
            if a1 is not None and "a1" in prev_a:
                S_id01 = _cos01(a1, prev_a["a1"])
                s_id_hist.append(S_id01)

            S_id_anchor = None
            if a1 is not None and mu_anchor is not None:
                S_id_anchor = _cos01(a1, mu_anchor.unsqueeze(0).expand_as(a1))

            welfare = None
            if a_crit is not None:
                welfare = float(critics(a_crit.detach())["aggregate"].mean().item())

            # ── CIT training step ──
            loss_c = torch.tensor(0.0, device=device)
            loss_preserve = torch.tensor(0.0, device=device)
            preserve_raw = 0.0
            grad_norm = 0.0
            did_update = False
            crit_sat = None

            if (
                "L_CIT" in losses_enabled
                and a_crit is not None
                and a_nat is not None
                and opt_heads is not None
                and lambda_cit_eff > 0.0
            ):
                # Check saturation guard
                with torch.no_grad():
                    agg_now = critics(a_crit.detach())["aggregate"]
                    crit_sat = float((agg_now > sat_threshold).float().mean().item())

                spread_val = _spread(a_crit.detach())

                if crit_sat is not None and crit_sat > stop_crit_sat:
                    # All samples already above threshold — stop training
                    opt_heads = None
                    print(f"  [step {step}] Critic saturation {crit_sat:.2f} "
                          f"> {stop_crit_sat} — freezing heads")
                else:
                    # CIT loss: push non-compliant samples toward compliance
                    loss_c = L_cit(a_crit, critics) * lambda_cit_eff

                    # Preserve loss: keep representations near natural state
                    # FIX: .mean(dim=-1) not .sum(dim=-1)
                    preserve_raw = float(
                        (a_crit - a_nat.detach()).pow(2).mean(dim=-1).mean().item()
                    )
                    loss_preserve = lambda_preserve * (
                        (a_crit - a_nat.detach()).pow(2).mean(dim=-1).mean()
                    )

                    loss_update = loss_c + loss_preserve

                    opt_heads.zero_grad(set_to_none=True)
                    loss_update.backward()

                    # Gradient clipping
                    if max_grad_norm > 0 and trainable_params:
                        grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                trainable_params, max_grad_norm
                            ).item()
                        )

                    opt_heads.step()
                    did_update = True

            # ── Log ──
            row = {
                "ts": _ts(),
                "arm": arm_id,
                "task_id": task.task_id,
                "step": step,
                "epoch": epoch,
                "phase": sched.phase,
                "S_id01": S_id01,
                "S_id_anchor": S_id_anchor,
                "welfare": welfare,
                "loss_cit": float(loss_c.item()),
                "loss_preserve": float(loss_preserve.item()),
                "preserve_raw": preserve_raw,
                "grad_norm": grad_norm,
                "did_update": did_update,
                "lambda_cit_eff": lambda_cit_eff,
                "crit_sat": crit_sat,
                "a_key": a_key,
            }
            _jsonl(log_path, row)

            prev_a = {k: v.detach() for k, v in out.items()}

            # ── Early stop ──
            if sched.should_early_stop(s_id_hist):
                print(f"  [step {step}] Early stop triggered")
                early_stopped = True

    print(f"[INFO] {arm_id}: training done ({step} steps, phase={sched.phase})")

    # ══════════════════════════════════════════════════════
    # POST-HOC EVALUATION
    # ══════════════════════════════════════════════════════
    # Clean measurement: evaluate ALL prompts with current
    # (trained) heads vs reference (initial) heads, B=1.
    # ══════════════════════════════════════════════════════

    print(f"[INFO] {arm_id}: post-hoc eval ({len(tasks)} prompts)...")

    ph_trained: List[float] = []
    ph_ref: List[float] = []
    ph_deltas: List[float] = []

    with torch.no_grad():
        for idx, task in enumerate(tasks):
            tok_e = model.tokenizer(
                [task.prompt], return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            ids_e = tok_e["input_ids"].to(device)
            mask_e = tok_e.get("attention_mask", torch.ones_like(ids_e)).to(device)

            # Forward with CURRENT (trained for A3, unchanged for A0) heads
            out_e, pooled_e = model(ids_e, mask_e, return_pooled=True)
            a_e = out_e.get(a_key)

            w_trained = float("nan")
            pr_trained: list = []
            if a_e is not None:
                cr = critics(a_e)
                w_trained = float(cr["aggregate"].item())
                pr_trained = cr["per_rule"].squeeze(0).tolist()

            # Forward with REFERENCE (initial) heads — reuse pooled states
            h_e = pooled_e.get(f"h{crit_head_idx + 1}")
            w_ref = w_trained  # fallback
            pr_ref = pr_trained
            if h_e is not None:
                a_ref = ref_probe_heads[crit_head_idx](h_e)
                cr_ref = critics(a_ref)
                w_ref = float(cr_ref["aggregate"].item())
                pr_ref = cr_ref["per_rule"].squeeze(0).tolist()

            delta = w_trained - w_ref
            ph_trained.append(w_trained)
            ph_ref.append(w_ref)
            ph_deltas.append(delta)

            _jsonl(posthoc_path, {
                "task_id": task.task_id,
                "idx": idx,
                "welfare_trained": round(w_trained, 6),
                "welfare_ref": round(w_ref, 6),
                "delta": round(delta, 6),
                "pr_trained": [round(x, 4) for x in pr_trained],
                "pr_ref": [round(x, 4) for x in pr_ref],
            })

    # ── Post-hoc summary ──
    n_total = len(ph_deltas)
    n_improved = sum(1 for d in ph_deltas if d > 0.001)
    n_degraded = sum(1 for d in ph_deltas if d < -0.001)
    n_neutral = n_total - n_improved - n_degraded

    lines = [
        f"arm: {arm_id}",
        f"random_critics: {use_random_critics}",
        f"seed: {seed}",
        f"steps: {step}",
        f"final_phase: {sched.phase}",
        f"early_stopped: {early_stopped}",
        f"",
        f"welfare_trained_avg: {statistics.mean(ph_trained):.6f}",
        f"welfare_trained_min: {min(ph_trained):.6f}",
        f"welfare_trained_max: {max(ph_trained):.6f}",
        f"",
        f"welfare_ref_avg: {statistics.mean(ph_ref):.6f}",
        f"welfare_ref_min: {min(ph_ref):.6f}",
        f"welfare_ref_max: {max(ph_ref):.6f}",
        f"",
        f"delta_avg: {statistics.mean(ph_deltas):.6f}",
        f"delta_min: {min(ph_deltas):.6f}",
        f"delta_max: {max(ph_deltas):.6f}",
        f"delta_stdev: {statistics.stdev(ph_deltas) if n_total > 1 else 0:.6f}",
        f"",
        f"n_improved: {n_improved}  (delta > +0.001)",
        f"n_degraded: {n_degraded}  (delta < -0.001)",
        f"n_neutral:  {n_neutral}",
    ]
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    for ln in lines:
        if ln:
            print(f"  {ln}")

    # ── Restore heads if using preloaded model ──
    if preloaded_model is not None:
        for i, sd in enumerate(initial_heads_sd):
            model.probe_heads[i].load_state_dict(sd)

    # ── Cleanup if we loaded our own model ──
    if own_model:
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "arm": arm_id,
        "delta_avg": statistics.mean(ph_deltas),
        "n_improved": n_improved,
        "n_degraded": n_degraded,
        "n_total": n_total,
    }


# ── CLI ───────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="CIT ablation experiment (v2)")
    ap.add_argument("--ablation", required=True, help="Ablation config YAML")
    ap.add_argument("--model", required=True, help="Model config YAML")
    ap.add_argument("--out", default="results/v2", help="Output directory")
    ap.add_argument("--arms", default="A0,A3", help="Comma-separated arm IDs")
    ap.add_argument("--anchor", default="", help="Path to anchor .pt")
    ap.add_argument("--heads", default="", help="Path to probe head checkpoint .pt")
    ap.add_argument("--critics", default="", help="Path to critics checkpoint .pt")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    ab_cfg = _yaml(Path(args.ablation))
    m_cfg = _yaml(Path(args.model))

    arm_ids = [x.strip().upper() for x in args.arms.split(",") if x.strip()]
    out = Path(args.out)
    _dir(out)

    for arm_id in arm_ids:
        run_arm(
            arm_id=arm_id,
            ab_cfg=ab_cfg,
            m_cfg=m_cfg,
            out_dir=out / arm_id,
            anchor_path=Path(args.anchor) if args.anchor else None,
            heads_path=Path(args.heads) if args.heads else None,
            critics_path=Path(args.critics) if args.critics else None,
            seed=args.seed,
        )

    print(f"\nDone. Results under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
