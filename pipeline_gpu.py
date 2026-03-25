"""
pipeline_gpu.py — Unified CIT pipeline for GPU (any model scale)
=================================================================

Replaces pipeline_70b.py.  Runs the full experimental pipeline:

  1. Load model (once)
  2. Compute anchor (mu_align)
  3. Train probe heads (L_self)
  4. Train critics on a3
  5. Run ablation: A0 + A3 + A3R for all shards

Fixes from pipeline_70b.py:
- AnchorStore: uses torch.save directly (no constructor mismatch)
- Memory: single model load, passed to all arms (no reloading)
- A3R arm: random critics sanity check built in

Usage (Gemma-4B on RTX 4090):
  python pipeline_gpu.py --config configs/gemma3_4b_gpu.yaml --seed 123

Usage (Qwen-72B on A100):
  python pipeline_gpu.py --config configs/qwen25_72b_gpu.yaml --seed 123

Usage (skip to ablation if artifacts exist):
  python pipeline_gpu.py --config configs/gemma3_4b_gpu.yaml --skip-to ablation
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.transformer_cit.model import CITModel
from src.transformer_cit.critics import CriticEnsemble
from src.transformer_cit.losses import CITLoss


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def cosine01(a, b):
    c = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return (c + 1.0) / 2.0


# ============================================================
# Step 1: Load model
# ============================================================

def load_model(cfg):
    """Load backbone + probe heads.  Returns (model, device_str)."""
    model_name = cfg["model"]["name"]
    tap_layers = list(cfg["model"]["tap_layers"])
    d = int(cfg["model"].get("identity_dim", 64))
    pooling = str(cfg["model"].get("pooling", "mean"))
    use_mlp = bool(cfg["model"].get("use_mlp_heads", False))
    use_ln = bool(cfg["model"].get("use_layernorm", True))
    q4 = bool(cfg["model"].get("quantize_4bit", False))
    dtype = str(cfg["model"].get("torch_dtype", "float32"))

    print(f"\n{'='*60}")
    print(f"  Loading model: {model_name}")
    print(f"  quantize_4bit={q4}, dtype={dtype}, layernorm={use_ln}")
    print(f"{'='*60}\n")

    model = CITModel(
        model_name=model_name,
        tap_layers=tap_layers,
        d=d,
        pooling=pooling,
        use_mlp_heads=use_mlp,
        use_layernorm=use_ln,
        quantize_4bit=q4,
        torch_dtype=dtype,
    )

    if q4:
        device = str(next(model.backbone.parameters()).device)
        for ph in model.probe_heads:
            ph.to(device)
    else:
        device = str(cfg.get("training", {}).get("device", "cuda"))
        model = model.to(device)

    model.eval()
    print(f"[INFO] Model on device: {device}")
    return model, device


# ============================================================
# Step 2: Compute anchor
# ============================================================

def compute_anchor(model, device, cfg, prompts_path, out_path):
    print(f"\n{'='*60}")
    print(f"  Step 2: Anchor from {prompts_path}")
    print(f"{'='*60}\n")

    if out_path.exists():
        print(f"[SKIP] Anchor exists: {out_path}")
        return

    max_len = int(cfg.get("cit", {}).get("max_length", 256))
    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])

    all_a1 = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            tok = model.tokenizer(
                [prompt], return_tensors="pt", padding=True,
                truncation=True, max_length=max_len,
            )
            ids = tok["input_ids"].to(device)
            mask = tok.get("attention_mask", torch.ones_like(ids)).to(device)
            out = model(input_ids=ids, attention_mask=mask)
            all_a1.append(out["a1"].detach().cpu())
            if (i + 1) % 5 == 0:
                print(f"  Anchor: {i+1}/{len(prompts)}")

    all_a1 = torch.cat(all_a1, dim=0)
    mu_align = all_a1.mean(dim=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Direct torch.save — no AnchorStore constructor needed
    torch.save({
        "mu_align": mu_align,
        "meta": {
            "n_prompts": len(prompts),
            "timestamp": now_ts(),
            "source": str(prompts_path),
        },
    }, str(out_path))
    print(f"[OK] Anchor saved: {out_path} (shape={mu_align.shape})")


# ============================================================
# Step 3: Train probe heads (L_self + L_id)
# ============================================================

def train_heads(model, device, cfg, prompts_path, anchor_path, out_path,
                epochs=50, seed=123):
    print(f"\n{'='*60}")
    print(f"  Step 3: Train probe heads ({epochs} epochs)")
    print(f"{'='*60}\n")

    if out_path.exists():
        print(f"[SKIP] Heads exist: {out_path}")
        return

    torch.manual_seed(seed)
    max_len = int(cfg.get("cit", {}).get("max_length", 256))

    # Load anchor
    payload = torch.load(str(anchor_path), weights_only=False)
    mu_align = payload["mu_align"].to(device)

    # Load prompts
    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])

    # Enable grad on probe heads
    for ph in model.probe_heads:
        for p in ph.parameters():
            p.requires_grad = True

    optimizer = torch.optim.Adam(
        [p for ph in model.probe_heads for p in ph.parameters()],
        lr=float(cfg.get("cit", {}).get("lr_heads", 1e-4)),
    )

    for epoch in range(epochs):
        total_loss = 0.0
        for prompt in prompts:
            tok = model.tokenizer(
                [prompt], return_tensors="pt", padding=True,
                truncation=True, max_length=max_len,
            )
            ids = tok["input_ids"].to(device)
            mask = tok.get("attention_mask", torch.ones_like(ids)).to(device)

            out = model(input_ids=ids, attention_mask=mask)
            a1 = out["a1"]
            loss = (a1 - mu_align.unsqueeze(0)).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                tok0 = model.tokenizer(
                    [prompts[0]], return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len,
                )
                ids0 = tok0["input_ids"].to(device)
                mask0 = tok0.get("attention_mask", torch.ones_like(ids0)).to(device)
                out0 = model(input_ids=ids0, attention_mask=mask0)
                s_id = float(cosine01(out0["a1"], mu_align.unsqueeze(0)).item())
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"loss={total_loss/len(prompts):.6f}, S_id={s_id:.4f}")

    # Freeze and save
    for ph in model.probe_heads:
        for p in ph.parameters():
            p.requires_grad = False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sd = [
        {k: v.detach().cpu().clone() for k, v in ph.state_dict().items()}
        for ph in model.probe_heads
    ]
    torch.save({
        "probe_heads": sd,
        "meta": {"epochs": epochs, "seed": seed, "timestamp": now_ts()},
    }, str(out_path))
    print(f"[OK] Heads saved: {out_path}")


# ============================================================
# Step 4: Train critics on a3
# ============================================================

def train_critics(model, device, cfg, calib_path, out_path,
                  a_key="a3", epochs=100, seed=123):
    print(f"\n{'='*60}")
    print(f"  Step 4: Train critics on {a_key} ({epochs} epochs)")
    print(f"{'='*60}\n")

    if out_path.exists():
        print(f"[SKIP] Critics exist: {out_path}")
        return

    torch.manual_seed(seed)
    max_len = int(cfg.get("cit", {}).get("max_length", 256))
    d = int(cfg["model"].get("identity_dim", 64))
    K = int(cfg.get("critics", {}).get("num_rules", 5))

    # Load calibration data
    calib_data = []
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                calib_data.append(json.loads(line))

    # Extract representations
    print(f"  Extracting {a_key} for {len(calib_data)} prompts...")
    reps, labels = [], []
    with torch.no_grad():
        for item in calib_data:
            tok = model.tokenizer(
                [item["prompt"]], return_tensors="pt", padding=True,
                truncation=True, max_length=max_len,
            )
            ids = tok["input_ids"].to(device)
            mask = tok.get("attention_mask", torch.ones_like(ids)).to(device)
            out = model(input_ids=ids, attention_mask=mask)
            a = out[a_key]
            reps.append(a.squeeze(0).cpu())
            labels.append(float(item["h_C"]))

    reps_t = torch.stack(reps).to(device)      # [N, d]
    labels_t = torch.tensor(labels).to(device)  # [N]

    # Train critics (BCE per rule, equal targets for simplicity)
    critics = CriticEnsemble(K=K, d=d).to(device)
    optimizer = torch.optim.Adam(critics.parameters(), lr=1e-3)

    for epoch in range(epochs):
        scores = critics(reps_t)
        agg = scores["aggregate"]
        loss = torch.nn.functional.binary_cross_entropy(agg, labels_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            with torch.no_grad():
                preds = (agg > 0.5).float()
                acc = (preds == labels_t).float().mean().item()
                try:
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(
                        labels_t.cpu().numpy(), agg.detach().cpu().numpy()
                    )
                except Exception:
                    auc = float("nan")
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"loss={loss.item():.4f}, acc={acc:.3f}, AUC={auc:.4f}")

    # Freeze and save
    critics.freeze()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": critics.state_dict(),
        "meta": {
            "a_key": a_key,
            "epochs": epochs,
            "seed": seed,
            "timestamp": now_ts(),
            "n_samples": len(calib_data),
        },
        "rule_names": cfg.get("critics", {}).get("rules", []),
    }, str(out_path))
    print(f"[OK] Critics saved: {out_path}")


# ============================================================
# Step 5: Ablation (all shards × A0 + A3 + A3R)
# ============================================================

def run_ablation(model, device, cfg, shard_dir, prompts_dir,
                 artifacts_dir, out_base, seed=123, arms=("A0", "A3", "A3R")):
    """Run ablation arms for all available shards."""
    print(f"\n{'='*60}")
    print(f"  Step 5: Ablation ({', '.join(arms)})")
    print(f"{'='*60}\n")

    from src.transformer_cit.run_experiment import run_arm

    anchor_path = artifacts_dir / "anchor_v2.pt"
    heads_path = artifacts_dir / "heads_v2.pt"
    critics_path = artifacts_dir / "critics_v2.pt"

    # Find or generate shard configs
    shard_configs = sorted(shard_dir.glob("ablation_step9_shard*_v2.yaml"))
    if not shard_configs:
        print("[INFO] Generating v2 shard configs...")
        generate_v2_shard_configs(cfg, shard_dir, prompts_dir)
        shard_configs = sorted(shard_dir.glob("ablation_step9_shard*_v2.yaml"))

    if not shard_configs:
        # Fallback: use optC configs if they exist
        shard_configs = sorted(shard_dir.glob("ablation_step9_shard*_optC.yaml"))

    print(f"  Found {len(shard_configs)} shard configs")

    results = []
    for sc in shard_configs:
        shard_name = sc.stem.replace("ablation_step9_", "").replace("_v2", "").replace("_optC", "")
        ab_cfg = load_yaml(sc)

        # Ensure A3R arm exists in config
        if "A3R" not in ab_cfg.get("ablation", {}).get("arms", {}):
            ab_cfg.setdefault("ablation", {}).setdefault("arms", {})["A3R"] = {
                "description": "CIT with random critics (sanity check)",
                "enable_probes": True,
                "losses": ["L_CIT"],
                "random_critics": True,
            }

        for arm_id in arms:
            arm_out = Path(out_base) / shard_name / arm_id
            summary_file = arm_out / f"{arm_id.lower()}_posthoc_summary.txt"
            if summary_file.exists():
                print(f"  [SKIP] {shard_name} {arm_id} already done")
                continue

            print(f"\n  --- {shard_name} {arm_id} ---")
            try:
                result = run_arm(
                    arm_id=arm_id,
                    ab_cfg=ab_cfg,
                    m_cfg=cfg,
                    out_dir=arm_out,
                    anchor_path=anchor_path,
                    heads_path=heads_path,
                    critics_path=critics_path,
                    seed=seed,
                    preloaded_model=model,
                    preloaded_device=device,
                )
                results.append({**result, "shard": shard_name})
            except Exception as e:
                print(f"  [ERROR] {shard_name} {arm_id}: {e}")
                import traceback
                traceback.print_exc()

    # Print final summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['shard']} {r['arm']}: delta_avg={r['delta_avg']:+.4f}, "
              f"improved={r['n_improved']}/{r['n_total']}, "
              f"degraded={r['n_degraded']}/{r['n_total']}")


def generate_v2_shard_configs(cfg, shard_dir, prompts_dir):
    """Generate v2 ablation configs for all shard promptpacks."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(prompts_dir.glob("critic_eval_v0_shard*.jsonl"))

    template = {
        "ablation": {
            "tau_welfare": 0.6,
            "cit_batch_size": 2,
            "lambda_preserve": 0.3,
            "sat_threshold": 0.95,
            "stop_cos_spread": 0.99,
            "stop_critic_saturation": 1.0,
            "max_grad_norm": 1.0,
            "arms": {
                "A0": {
                    "description": "Baseline (probes ON), no training",
                    "enable_probes": True,
                    "losses": [],
                },
                "A3": {
                    "description": "Full CIT training",
                    "enable_probes": True,
                    "losses": ["L_CIT"],
                },
                "A3R": {
                    "description": "CIT with random critics (sanity check)",
                    "enable_probes": True,
                    "losses": ["L_CIT"],
                    "random_critics": True,
                },
            },
        },
        "rollout_steps": 3,
    }

    for sf in shard_files:
        shard_id = sf.stem.replace("critic_eval_v0_", "")
        t = yaml.safe_load(yaml.dump(template))  # deep copy
        t["ablation"]["promptpack"] = str(sf)

        out_path = shard_dir / f"ablation_step9_{shard_id}_v2.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(t, f, default_flow_style=False)
        print(f"    [OK] {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="CIT Pipeline (GPU, any scale)")
    ap.add_argument("--config", required=True, help="Model config YAML")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--skip-to",
                    choices=["anchor", "heads", "critics", "ablation"],
                    default=None)
    ap.add_argument("--arms", default="A0,A3,A3R",
                    help="Arms to run (comma-separated)")
    ap.add_argument("--hf-token", default="", help="HuggingFace token")
    args = ap.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    cfg = load_yaml(Path(args.config))

    # Determine artifact directory from model name
    model_short = cfg["model"]["name"].split("/")[-1].lower().replace("-", "_")
    artifacts = Path(f"artifacts/{model_short}")
    artifacts.mkdir(parents=True, exist_ok=True)

    anchor_path = artifacts / "anchor_v2.pt"
    heads_path = artifacts / "heads_v2.pt"
    critics_path = artifacts / "critics_v2.pt"

    # ── Step 1: Load model ──
    model, device = load_model(cfg)

    # ── Step 2: Anchor ──
    if args.skip_to not in ["heads", "critics", "ablation"]:
        compute_anchor(
            model, device, cfg,
            prompts_path=Path("prompts/anchor_prompts_v0.jsonl"),
            out_path=anchor_path,
        )

    # ── Step 3: Train heads ──
    if args.skip_to not in ["critics", "ablation"]:
        train_heads(
            model, device, cfg,
            prompts_path=Path("prompts/anchor_prompts_v0.jsonl"),
            anchor_path=anchor_path,
            out_path=heads_path,
            epochs=50,
            seed=args.seed,
        )
        # Reload trained heads into model
        ckpt = torch.load(str(heads_path), map_location=device, weights_only=False)
        for i, sd in enumerate(ckpt.get("probe_heads", [])):
            if i < len(model.probe_heads):
                model.probe_heads[i].load_state_dict(sd)
        print(f"[INFO] Reloaded trained heads into model")

    # ── Step 4: Train critics ──
    if args.skip_to not in ["ablation"]:
        train_critics(
            model, device, cfg,
            calib_path=Path("prompts/critic_calib_v0.jsonl"),
            out_path=critics_path,
            a_key="a3",
            epochs=100,
            seed=args.seed,
        )

    # If skipping to ablation, load trained heads into model
    if args.skip_to == "ablation" and heads_path.exists():
        ckpt = torch.load(str(heads_path), map_location=device, weights_only=False)
        for i, sd in enumerate(ckpt.get("probe_heads", [])):
            if i < len(model.probe_heads):
                model.probe_heads[i].load_state_dict(sd)
        print(f"[INFO] Loaded trained heads for ablation")

    # ── Step 5: Ablation ──
    shard_dir = Path("configs/step9_shards")
    prompts_dir = Path("prompts/step9_shards")
    out_base = Path(f"results/v2/seed{args.seed}")

    arm_list = tuple(a.strip().upper() for a in args.arms.split(",") if a.strip())

    run_ablation(
        model, device, cfg,
        shard_dir, prompts_dir, artifacts,
        out_base, seed=args.seed, arms=arm_list,
    )

    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  Results: {out_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
