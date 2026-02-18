"""
Anchor Offline (Phase 2): compute mu_align from frozen backbone + frozen critics.

Procedure (spec v1 §4 Phase 2):
  1. Load frozen backbone + probe heads (CITModel)
  2. Load frozen critics (CriticEnsemble)
  3. For each "good state" prompt:
     a. Extract a^(1)_base via forward pass
     b. Compute revision target via gradient ascent on s_R
     c. a1_revised = a1_rev + epsilon * grad
  4. mu_align = mean(a1_revised over all prompts)
  5. Save mu_align + metadata to .pt file

Usage:
  # Dry (no download):
  python -m src.transformer_cit.anchor_offline --dry --out artifacts/anchor_v0.pt

  # Real (downloads backbone, CPU):
  python -m src.transformer_cit.anchor_offline \\
      --model configs/gemma3_4b_cpu.yaml \\
      --prompts prompts/anchor_prompts_v0.jsonl \\
      --out artifacts/anchor_v0.pt
"""
import argparse
import json
import os
import sys
import time

import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        print("[WARN] PyYAML not installed. Using hardcoded defaults.")
        return _default_config()


def _default_config() -> dict:
    return {
        "model": {
            "name": "google/gemma-3-4b-it",
            "tap_layers": [11, 22, 33],
            "identity_dim": 64,
            "pooling": "mean",
            "use_mlp_heads": False,
        },
        "critics": {"num_rules": 5, "hidden_dim": 64},
        "anchor": {"revision_epsilon": 0.01},
    }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


# ---------------------------------------------------------------------------
# Revision step (shared by dry + real)
# ---------------------------------------------------------------------------

def revise_single(a1: torch.Tensor, critics, epsilon: float) -> torch.Tensor:
    """Apply one revision step: gradient ascent on s_R w.r.t. a1.

    Args:
        a1: [1, d] detached base representation
        critics: frozen CriticEnsemble
        epsilon: step size
    Returns:
        a1_revised: [1, d] detached revised representation
    """
    a1_rev = a1.detach().requires_grad_(True)
    result = critics(a1_rev)
    s_R = result["aggregate"]  # [1]
    grad = torch.autograd.grad(s_R.sum(), a1_rev)[0]
    a1_revised = a1_rev + epsilon * grad  # [1, d]
    return a1_revised.detach(), s_R.item()


# ---------------------------------------------------------------------------
# Dry mode
# ---------------------------------------------------------------------------

def run_dry(d: int, K: int, n_prompts: int, epsilon: float,
            out_path: str) -> None:
    from .critics import CriticEnsemble

    print(f"[DRY] identity_dim={d}, num_rules={K}, n_prompts={n_prompts}")

    torch.manual_seed(42)
    a1_base = torch.randn(n_prompts, d)

    critics = CriticEnsemble(K=K, d=d, frozen=True)
    print(f"[DRY] Critics frozen: {not any(p.requires_grad for p in critics.parameters())}")

    revised_list = []
    for i in range(n_prompts):
        a1_i = a1_base[i].unsqueeze(0)
        a1_rev, s_R = revise_single(a1_i, critics, epsilon)
        revised_list.append(a1_rev)
        if i == 0 or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_prompts}] s_R={s_R:.4f}")

    mu_align = torch.cat(revised_list, dim=0).mean(dim=0)

    print(f"[DRY] mu_align shape: {mu_align.shape}")
    print(f"[DRY] mu_align norm:  {mu_align.norm().item():.4f}")

    _save_anchor(mu_align, out_path, meta={
        "mode": "dry",
        "identity_dim": d,
        "num_rules": K,
        "n_samples": n_prompts,
        "revision_epsilon": epsilon,
    })


# ---------------------------------------------------------------------------
# Real mode
# ---------------------------------------------------------------------------

def run_real(config: dict, prompts: list[dict], out_path: str) -> None:
    from .model import CITModel
    from .critics import CriticEnsemble

    mc = config["model"]
    cc = config["critics"]
    ac = config["anchor"]

    model_name = mc["name"]
    tap_layers = mc["tap_layers"]
    d = mc["identity_dim"]
    pooling = mc["pooling"]
    use_mlp = mc.get("use_mlp_heads", False)
    K = cc["num_rules"]
    epsilon = ac["revision_epsilon"]

    print(f"[REAL] Loading backbone: {model_name}")
    print(f"[REAL] Taps: {tap_layers}, d={d}, pooling={pooling}")

    model = CITModel(
        model_name=model_name, tap_layers=tap_layers,
        d=d, pooling=pooling, use_mlp_heads=use_mlp,
    )
    model.eval()

    critics = CriticEnsemble(K=K, d=d, frozen=True)
    print(f"[REAL] Critics frozen: {not any(p.requires_grad for p in critics.parameters())}")

    tokenizer = model.tokenizer
    prompt_texts = [p["prompt"] for p in prompts]
    print(f"[REAL] Processing {len(prompt_texts)} prompts...")

    revised_list = []
    for i, text in enumerate(prompt_texts):
        tokens = tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )

        with torch.no_grad():
            identity = model(tokens["input_ids"], tokens["attention_mask"])
        a1 = identity["a1"]  # [1, d]

        a1_rev, s_R = revise_single(a1, critics, epsilon)
        revised_list.append(a1_rev)

        if i == 0 or (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(prompt_texts)}] s_R={s_R:.4f}")

    mu_align = torch.cat(revised_list, dim=0).mean(dim=0)

    print(f"[REAL] mu_align shape: {mu_align.shape}")
    print(f"[REAL] mu_align norm:  {mu_align.norm().item():.4f}")

    _save_anchor(mu_align, out_path, meta={
        "mode": "real",
        "model_name": model_name,
        "tap_layers": tap_layers,
        "identity_dim": d,
        "pooling": pooling,
        "num_rules": K,
        "n_samples": len(prompt_texts),
        "revision_epsilon": epsilon,
    })


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def _save_anchor(mu_align: torch.Tensor, out_path: str, meta: dict) -> None:
    meta["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"mu_align": mu_align, "meta": meta}, out_path)

    print(f"[SAVED] {out_path}")
    print(f"  mu_align: [{mu_align.shape[0]}], norm={mu_align.norm().item():.4f}")
    for k, v in meta.items():
        print(f"  {k}: {v}")


def load_anchor(path: str) -> dict:
    """Load anchor .pt file. Returns {'mu_align': tensor, 'meta': dict}."""
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CIT Anchor Offline — compute mu_align (Phase 2)",
    )
    parser.add_argument("--dry", action="store_true",
                        help="Dry run with random tensors (no model download)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model YAML config")
    parser.add_argument("--prompts", type=str,
                        default="prompts/anchor_prompts_v0.jsonl",
                        help="Path to JSONL prompts file")
    parser.add_argument("--out", type=str,
                        default="artifacts/anchor_v0.pt",
                        help="Output path for anchor .pt file")
    parser.add_argument("--n-dry-prompts", type=int, default=20,
                        help="Number of synthetic prompts in dry mode")

    args = parser.parse_args()

    if args.dry:
        cfg = _default_config()
        run_dry(
            d=cfg["model"]["identity_dim"],
            K=cfg["critics"]["num_rules"],
            n_prompts=args.n_dry_prompts,
            epsilon=cfg["anchor"]["revision_epsilon"],
            out_path=args.out,
        )
    else:
        if args.model is None:
            parser.error("--model is required in real mode (or use --dry)")
        config = load_config(args.model)
        prompts = load_prompts(args.prompts)
        if len(prompts) == 0:
            print("[ERROR] No prompts loaded. Check --prompts path.")
            sys.exit(1)
        print(f"[INFO] Loaded {len(prompts)} prompts from {args.prompts}")
        run_real(config, prompts, args.out)


if __name__ == "__main__":
    main()
