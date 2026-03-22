"""
pipeline_70b.py — Complete CIT pipeline for Llama-3.1-70B on GPU
================================================================
Runs the FULL pipeline in a single process to avoid reloading
the 70B model multiple times:
  1. Load model (once)
  2. Compute anchor (mu_align)
  3. Train probe heads (L_self + L_id)
  4. Train critics on a3
  5. Run Phase 1 ablation (12 shards × A0 + A3)

Usage (from repo root):
  python pipeline_70b.py --config configs/llama31_70b_gpu.yaml --hf-token YOUR_TOKEN

Requires: A100 80GB (or equivalent), bitsandbytes, accelerate
"""

from __future__ import annotations

import argparse
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
from src.transformer_cit.anchor import AnchorStore


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
    model_name = cfg["model"]["name"]
    tap_layers = list(cfg["model"]["tap_layers"])
    d = int(cfg["model"].get("identity_dim", 64))
    pooling = str(cfg["model"].get("pooling", "mean"))
    use_mlp = bool(cfg["model"].get("use_mlp_heads", False))
    q4 = bool(cfg["model"].get("quantize_4bit", False))
    dtype = str(cfg["model"].get("torch_dtype", "float32"))

    print(f"\n{'='*60}")
    print(f"  Loading model: {model_name}")
    print(f"  Quantize 4-bit: {q4}, dtype: {dtype}")
    print(f"  Tap layers: {tap_layers}, d={d}")
    print(f"{'='*60}\n")

    model = CITModel(
        model_name=model_name,
        tap_layers=tap_layers,
        d=d,
        pooling=pooling,
        use_mlp_heads=use_mlp,
        quantize_4bit=q4,
        torch_dtype=dtype,
    )

    if q4:
        device = str(next(model.backbone.parameters()).device)
        for ph in model.probe_heads:
            ph.to(device)
    else:
        device = str(cfg.get("training", {}).get("device", "cpu"))
        model = model.to(device)

    model.eval()
    print(f"[INFO] Model loaded on device: {device}")
    return model, device


# ============================================================
# Step 2: Compute anchor
# ============================================================
def compute_anchor(model, device, cfg, prompts_path, out_path):
    print(f"\n{'='*60}")
    print(f"  Step 2: Computing anchor from {prompts_path}")
    print(f"{'='*60}\n")

    if out_path.exists():
        print(f"[SKIP] Anchor already exists: {out_path}")
        return

    max_len = int(cfg.get("cit", {}).get("max_length", 256))
    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                prompts.append(obj["prompt"])

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
            a1 = out["a1"].detach().cpu()  # [1, d]
            all_a1.append(a1)
            if (i + 1) % 5 == 0:
                print(f"  Anchor: {i+1}/{len(prompts)} prompts")

    all_a1 = torch.cat(all_a1, dim=0)  # [N, d]
    mu_align = all_a1.mean(dim=0)       # [d]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    store = AnchorStore(mu_align=mu_align, meta={
        "n_prompts": len(prompts),
        "timestamp": now_ts(),
        "source": str(prompts_path),
    })
    store.save(str(out_path))
    print(f"[OK] Anchor saved: {out_path} (mu_align shape: {mu_align.shape})")


# ============================================================
# Step 3: Train probe heads (L_self)
# ============================================================
def train_heads(model, device, cfg, prompts_path, anchor_path, out_path, epochs=50, seed=123):
    print(f"\n{'='*60}")
    print(f"  Step 3: Training probe heads ({epochs} epochs)")
    print(f"{'='*60}\n")

    if out_path.exists():
        print(f"[SKIP] Heads already exist: {out_path}")
        return

    torch.manual_seed(seed)
    max_len = int(cfg.get("cit", {}).get("max_length", 256))

    # Load anchor
    store = AnchorStore.load_from_file(str(anchor_path))
    mu_align = store.mu_align.to(device)

    # Load prompts
    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                prompts.append(obj["prompt"])

    # Enable grad on probe heads only
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
            a1 = out["a1"]  # [1, d]

            # L_self: pull toward anchor
            loss = (a1 - mu_align.unsqueeze(0)).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(prompts)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Check S_id
            with torch.no_grad():
                tok0 = model.tokenizer(
                    [prompts[0]], return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len,
                )
                ids0 = tok0["input_ids"].to(device)
                mask0 = tok0.get("attention_mask", torch.ones_like(ids0)).to(device)
                out0 = model(input_ids=ids0, attention_mask=mask0)
                s_id = float(cosine01(out0["a1"], mu_align.unsqueeze(0)).item())
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg:.6f}, S_id={s_id:.4f}")

    # Freeze heads again
    for ph in model.probe_heads:
        for p in ph.parameters():
            p.requires_grad = False

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sd = [
        {k: v.detach().cpu().clone() for k, v in ph.state_dict().items()}
        for ph in model.probe_heads
    ]
    torch.save({"probe_heads": sd, "meta": {
        "epochs": epochs, "seed": seed, "timestamp": now_ts(),
    }}, str(out_path))
    print(f"[OK] Heads saved: {out_path}")


# ============================================================
# Step 4: Train critics on a3
# ============================================================
def train_critics(model, device, cfg, calib_path, out_path,
                  a_key="a3", epochs=100, seed=123):
    print(f"\n{'='*60}")
    print(f"  Step 4: Training critics on {a_key} ({epochs} epochs)")
    print(f"{'='*60}\n")

    if out_path.exists():
        print(f"[SKIP] Critics already exist: {out_path}")
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
                obj = json.loads(line)
                calib_data.append(obj)

    # Extract representations
    print(f"  Extracting {a_key} representations for {len(calib_data)} prompts...")
    reps = []
    labels = []
    with torch.no_grad():
        for i, item in enumerate(calib_data):
            tok = model.tokenizer(
                [item["prompt"]], return_tensors="pt", padding=True,
                truncation=True, max_length=max_len,
            )
            ids = tok["input_ids"].to(device)
            mask = tok.get("attention_mask", torch.ones_like(ids)).to(device)
            out = model(input_ids=ids, attention_mask=mask)
            a = out[a_key].detach()  # [1, d]
            reps.append(a)
            labels.append(float(item.get("h_C", 1)))

    reps_t = torch.cat(reps, dim=0)  # [N, d]
    labels_t = torch.tensor(labels, device=device).unsqueeze(-1)  # [N, 1]

    # Train critics
    critics = CriticEnsemble(K=K, d=d).to(device)
    optimizer = torch.optim.Adam(critics.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(epochs):
        out_c = critics(reps_t)
        agg = out_c["aggregate"].unsqueeze(-1)  # [N, 1]
        loss = loss_fn(agg, labels_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            with torch.no_grad():
                preds = (agg > 0.5).float()
                acc = (preds == labels_t).float().mean().item()
                # AUC approximation
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(
                        labels_t.cpu().numpy(),
                        agg.detach().cpu().numpy()
                    )
                except:
                    auc = float("nan")
            print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}, acc={acc:.3f}, AUC={auc:.4f}")

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

    return critics


# ============================================================
# Step 5: Run ablation (Phase 1)
# ============================================================
def run_phase1(model, device, cfg, shard_dir, prompts_dir, artifacts_dir,
               out_base, seed=123):
    """Run A0 + A3 for all available shards."""
    print(f"\n{'='*60}")
    print(f"  Step 5: Phase 1 ablation")
    print(f"{'='*60}\n")

    # This delegates to the existing run_ablation module
    # We import and call run_arm directly
    from src.transformer_cit.run_ablation import run_arm, load_yaml as _ly
    from src.transformer_cit.run_ablation import load_promptpack, PromptTask

    # Load artifacts
    anchor_path = artifacts_dir / "anchor_real_v0.pt"
    heads_path = artifacts_dir / "heads_lself_lid_seed123_50.pt"
    critics_path = artifacts_dir / "critics_calib_a3_seed123_100.pt"

    # Find all shard configs
    shard_configs = sorted(shard_dir.glob("ablation_step9_shard*_optC.yaml"))
    if not shard_configs:
        print("[WARN] No optC shard configs found. Generating...")
        generate_shard_configs(cfg, shard_dir, prompts_dir)
        shard_configs = sorted(shard_dir.glob("ablation_step9_shard*_optC.yaml"))

    model_config_path = None  # We'll pass the cfg dict directly

    print(f"  Found {len(shard_configs)} shard configs")

    for sc in shard_configs:
        shard_name = sc.stem.replace("ablation_step9_", "").replace("_optC", "")
        ab_cfg = load_yaml(sc)

        for arm_id in ["A0", "A3"]:
            arm_out = Path(out_base) / shard_name / arm_id
            summary_file = arm_out / f"{arm_id.lower()}_posthoc_summary.txt"
            if summary_file.exists():
                print(f"  [SKIP] {shard_name} {arm_id} already done")
                continue

            print(f"\n  --- {shard_name} {arm_id} ---")
            try:
                run_arm(
                    arm_id=arm_id,
                    ab_cfg=ab_cfg,
                    m_cfg=cfg,
                    out_dir=arm_out,
                    anchor_path=anchor_path,
                    heads_path=heads_path,
                    critics_path=critics_path,
                    dry=False,
                    seed=seed,
                )
            except Exception as e:
                print(f"  [ERROR] {shard_name} {arm_id}: {e}")
                continue


def generate_shard_configs(cfg, shard_dir, prompts_dir):
    """Generate optC configs for all available shard promptpacks."""
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
                    "description": "Baseline (probes ON), no training updates",
                    "enable_probes": True,
                    "losses": [],
                    "schedule": "forge_anchor_preserve",
                },
                "A3": {
                    "description": "CIT on a3 (heads-only) + preserve + guardrails",
                    "enable_probes": True,
                    "losses": ["L_CIT"],
                    "schedule": "forge_anchor_preserve",
                },
            },
            "seeds": [123],
        },
        "rollout_steps": 3,
    }

    for sf in shard_files:
        shard_id = sf.stem.replace("critic_eval_v0_", "")
        t = dict(template)
        t["ablation"] = dict(template["ablation"])
        t["ablation"]["promptpack"] = str(sf)

        out_path = shard_dir / f"ablation_step9_{shard_id}_optC.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(t, f, default_flow_style=False)
        print(f"    [OK] {out_path}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="CIT Pipeline for Llama-3.1-70B")
    ap.add_argument("--config", default="configs/llama31_70b_gpu.yaml",
                    help="Model config YAML")
    ap.add_argument("--hf-token", default="", help="HuggingFace token for Llama access")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--skip-to", choices=["anchor", "heads", "critics", "ablation"],
                    default=None, help="Skip to a specific step (previous artifacts must exist)")
    args = ap.parse_args()

    # HuggingFace auth
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        print(f"[INFO] HF_TOKEN set")

    cfg = load_yaml(Path(args.config))
    artifacts = Path("artifacts/llama70b")
    artifacts.mkdir(parents=True, exist_ok=True)

    anchor_path = artifacts / "anchor_real_v0.pt"
    heads_path = artifacts / "heads_lself_lid_seed123_50.pt"
    critics_path = artifacts / "critics_calib_a3_seed123_100.pt"

    # Load model ONCE
    model, device = load_model(cfg)

    # Step 2: Anchor
    if args.skip_to not in ["heads", "critics", "ablation"]:
        compute_anchor(
            model, device, cfg,
            prompts_path=Path("prompts/anchor_prompts_v0.jsonl"),
            out_path=anchor_path,
        )

    # Step 3: Train heads
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
        ckpt = torch.load(str(heads_path), map_location=device)
        heads_sd = ckpt.get("probe_heads", [])
        n = min(len(model.probe_heads), len(heads_sd))
        for i in range(n):
            model.probe_heads[i].load_state_dict(heads_sd[i])
        print(f"[INFO] Reloaded {n} trained heads into model")

    # Step 4: Train critics
    if args.skip_to not in ["ablation"]:
        train_critics(
            model, device, cfg,
            calib_path=Path("prompts/critic_calib_v0.jsonl"),
            out_path=critics_path,
            a_key="a3",
            epochs=100,
            seed=args.seed,
        )

    # Step 5: Phase 1 ablation
    print("\n[INFO] Starting Phase 1 ablation...")
    print("[INFO] NOTE: Phase 1 uses run_ablation.run_arm which reloads model per arm.")
    print("[INFO] For 70B this is expensive. Consider running shards in batches.\n")

    # For the ablation, we need to use the existing runner
    # which handles the full training loop with post-hoc eval.
    # The runner will reload the model — unavoidable with current architecture.
    # But artifacts (anchor, heads, critics) are already computed.

    shard_dir = Path("configs/step9_shards")
    prompts_dir = Path("prompts/step9_shards")
    out_base = Path(f"results/step9_optC_70b/seed{args.seed}")

    run_phase1(model, device, cfg, shard_dir, prompts_dir, artifacts,
               out_base, seed=args.seed)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  Results: {out_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
