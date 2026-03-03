from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

# Ensure repo root is on sys.path so we can import src.transformer_cit.*
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.transformer_cit.model import CITModel  # noqa: E402
from src.transformer_cit.anchor import AnchorStore  # noqa: E402

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("PyYAML is required (yaml). Please install it in your venv.") from e


RULES = [
    "avoid_physical_harm",
    "avoid_deception",
    "respect_autonomy",
    "promote_helpfulness",
    "maintain_honesty",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


def auc_rank(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    Rank-based AUC (Mann–Whitney U) without sklearn.
    y_true: [N] in {0,1}
    y_score: [N] float
    """
    y_true = y_true.detach().cpu().float().view(-1)
    y_score = y_score.detach().cpu().float().view(-1)
    n = y_true.numel()
    if n == 0:
        return float("nan")
    pos = y_true >= 0.5
    n_pos = int(pos.sum().item())
    n_neg = int(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(y_score, stable=True)
    s = y_score[order]
    ranks = torch.zeros_like(s)

    i = 0
    rank = 1.0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1].item() == s[i].item():
            j += 1
        avg = (rank + (rank + (j - i))) / 2.0
        ranks[i : j + 1] = avg
        rank += (j - i + 1)
        i = j + 1

    ranks_full = torch.zeros_like(y_score)
    ranks_full[order] = ranks

    sum_ranks_pos = ranks_full[pos].sum().item()
    u = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(u / (n_pos * n_neg))


def summarize(X: torch.Tensor, y: torch.Tensor, name: str) -> None:
    """
    X: [N, d]
    y: [N] in {0,1}
    """
    X = X.detach().cpu()
    y = y.detach().cpu().float()

    N, d = X.shape
    mean = X.mean(dim=0)
    std = X.std(dim=0, unbiased=False)

    # Cosine to global mean direction
    m = mean / (mean.norm() + 1e-12)
    cos_to_mean = (X @ m) / (X.norm(dim=1) + 1e-12)

    pos = y >= 0.5
    neg = ~pos

    print(f"\n=== {name} ===")
    print(f"N={N}, d={d}")
    print(f"std_mean={std.mean().item():.6e}  std_min={std.min().item():.6e}  std_max={std.max().item():.6e}")
    print(f"norm_mean={X.norm(dim=1).mean().item():.6e}  norm_std={X.norm(dim=1).std(unbiased=False).item():.6e}")
    print(
        f"cos_to_mean: mean={cos_to_mean.mean().item():.6f} std={cos_to_mean.std(unbiased=False).item():.6f} "
        f"min={cos_to_mean.min().item():.6f} max={cos_to_mean.max().item():.6f}"
    )

    if pos.sum() > 0 and neg.sum() > 0:
        mu_pos = X[pos].mean(dim=0)
        mu_neg = X[neg].mean(dim=0)
        cos_mu = cosine(mu_pos, mu_neg).item()

        # Simple linear score along mean-difference direction
        w = (mu_pos - mu_neg)
        score = X @ w
        auc_lin = auc_rank(y, score)

        # Distance between class means
        dist = (mu_pos - mu_neg).norm().item()

        print(f"class_counts: pos={int(pos.sum())} neg={int(neg.sum())}")
        print(f"cos(mu_pos, mu_neg)={cos_mu:.6f}  ||mu_pos-mu_neg||={dist:.6e}")
        print(f"AUC(score = X·(mu_pos-mu_neg)) = {auc_lin:.4f}")
    else:
        print("Not enough class diversity to compute class stats/AUC.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/gemma3_4b_cpu.yaml")
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--data", default="prompts/critic_calib_v0.jsonl")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default=None, help="Override device (default from yaml)")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.model))
    device = args.device or str(cfg.get("training", {}).get("device", "cpu"))
    model_name = cfg["model"]["name"]
    tap_layers = list(cfg["model"].get("tap_layers", []))
    pooling = str(cfg["model"].get("pooling", "mean"))
    use_mlp = bool(cfg["model"].get("use_mlp_heads", False))

    store = AnchorStore.load_from_file(args.anchor)
    d = int(store.mu_align.shape[0])

    model = CITModel(
        model_name=model_name,
        tap_layers=tap_layers,
        d=d,
        pooling=pooling,
        use_mlp_heads=use_mlp,
    ).to(device)

    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    heads_path = Path(args.heads)
    ckpt = torch.load(str(heads_path), map_location=device)
    heads_sd = ckpt.get("probe_heads", [])
    if hasattr(model, "probe_heads") and heads_sd:
        n = min(len(model.probe_heads), len(heads_sd))
        for i in range(n):
            model.probe_heads[i].load_state_dict(heads_sd[i])
        print(f"[INFO] Loaded {n} probe head(s) from: {heads_path}")
    else:
        raise RuntimeError(f"No probe_heads found in checkpoint or model: {heads_path}")

    rows = load_jsonl(Path(args.data))
    y = torch.tensor([float(r.get("h_C", 1.0)) for r in rows], dtype=torch.float32)

    feats: Dict[str, List[torch.Tensor]] = {"a1": [], "a2": [], "a3": []}

    with torch.no_grad():
        for r in rows:
            prompt = r["prompt"]
            tok = model.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                padding=False,
            )
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            for k in list(feats.keys()):
                if k in out:
                    feats[k].append(out[k].squeeze(0).detach().cpu())

    for k, lst in feats.items():
        if not lst:
            print(f"\n[WARN] {k} not present (tap_layers may be < 3).")
            continue
        X = torch.stack(lst, dim=0)  # [N, d]
        summarize(X, y, k)


if __name__ == "__main__":
    main()