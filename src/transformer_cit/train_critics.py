"""
Stage 0: Calibrate constitutional critics C_k on a^(1)/a^(2)/a^(3).

- Loads frozen CITModel + probe heads checkpoint (train_heads.py format)
- Builds CriticEnsemble(K=5) operating on a_key in R^d
- Trains per-rule targets derived from violated_rules
- Computes AUC gate on aggregate score vs h_C (no sklearn)
- Saves critics checkpoint with meta
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .model import CITModel
from .critics import CriticEnsemble
from .anchor import AnchorStore

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required (yaml). Please install it in your venv.") from e


RULES: List[str] = [
    "avoid_physical_harm",
    "avoid_deception",
    "respect_autonomy",
    "promote_helpfulness",
    "maintain_honesty",
]


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_calib_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def targets_from_violations(violated_rules: List[str]) -> torch.Tensor:
    """
    Per-rule target: t_k = 1 if rule is satisfied, 0 if rule is violated.
    """
    vset = set(violated_rules or [])
    t = [0.0 if r in vset else 1.0 for r in RULES]
    return torch.tensor(t, dtype=torch.float32)  # [K]


def auc_rank(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    Rank-based AUC (Mann–Whitney U) without sklearn.
    y_true: [N] in {0,1}
    y_score: [N] float
    """
    y_true = y_true.detach().cpu().float().view(-1)
    y_score = y_score.detach().cpu().float().view(-1)
    assert y_true.numel() == y_score.numel()

    n = y_true.numel()
    if n == 0:
        return float("nan")

    pos_mask = (y_true >= 0.5)
    n_pos = int(pos_mask.sum().item())
    n_neg = int(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # ranks of scores (average rank for ties)
    order = torch.argsort(y_score, stable=True)
    scores_sorted = y_score[order]
    ranks = torch.zeros_like(scores_sorted)

    i = 0
    rank = 1.0
    while i < n:
        j = i
        while j + 1 < n and scores_sorted[j + 1].item() == scores_sorted[i].item():
            j += 1
        avg = (rank + (rank + (j - i))) / 2.0
        ranks[i : j + 1] = avg
        rank += (j - i + 1)
        i = j + 1

    ranks_full = torch.zeros_like(y_score)
    ranks_full[order] = ranks

    sum_ranks_pos = ranks_full[pos_mask].sum().item()
    u = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    auc = u / (n_pos * n_neg)
    return float(auc)


@dataclass
class Meta:
    timestamp: str
    a_key: str
    model: str
    device: str
    tap_layers: List[int]
    pooling: str
    use_mlp_heads: bool
    d: int
    K: int
    rules: List[str]
    steps: int
    lr: float
    seed: int
    max_length: int
    anchor: str
    heads: str
    data: str
    auc_aggregate: float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/gemma3_4b_cpu.yaml", help="Model YAML config")
    ap.add_argument("--anchor", required=True, help="Anchor .pt path (mu_align)")
    ap.add_argument("--heads", required=True, help="Probe-head checkpoint (.pt) from train_heads.py")
    ap.add_argument("--data", default="prompts/critic_calib_v0.jsonl", help="Calibration JSONL")
    ap.add_argument("--steps", type=int, default=100, help="Training steps for critics")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate for critics")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--max_length", type=int, default=256, help="Tokenizer max_length")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for critic training")
    ap.add_argument("--a_key", choices=["a1", "a2", "a3"], default="a1",
                    help="Which concentric vector to feed critics")
    ap.add_argument("--out", default="results/train_critics_v0", help="Output dir for logs")
    ap.add_argument("--save", default="artifacts/critics_calib_v0.pt", help="Save path for critics checkpoint")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = load_yaml(Path(args.model))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    # Load anchor (only to discover d)
    store = AnchorStore.load_from_file(args.anchor)
    mu = store.mu_align.detach()
    d = int(mu.shape[0])

    device = str(cfg.get("training", {}).get("device", "cpu"))
    model_name = cfg["model"]["name"]
    tap_layers = list(cfg["model"].get("tap_layers", []))
    pooling = str(cfg["model"].get("pooling", "mean"))
    use_mlp = bool(cfg["model"].get("use_mlp_heads", False))

    # Build model (frozen feature extractor)
    model = CITModel(
        model_name=model_name,
        tap_layers=tap_layers,
        d=d,
        pooling=pooling,
        use_mlp_heads=use_mlp,
    ).to(device)

    # Pad token safeguard
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    # Freeze EVERYTHING in model (backbone + probe heads)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load trained probe heads (pattern copied from run_ablation.py)
    heads_path = Path(args.heads)
    if not heads_path.exists():
        raise FileNotFoundError(f"Missing heads checkpoint: {heads_path}")

    ckpt = torch.load(str(heads_path), map_location=device)
    heads_sd = ckpt.get("probe_heads", [])
    if hasattr(model, "probe_heads") and heads_sd:
        n = min(len(model.probe_heads), len(heads_sd))
        for i in range(n):
            model.probe_heads[i].load_state_dict(heads_sd[i])
        print(f"[INFO] Loaded {n} probe head(s) from: {heads_path}")
    else:
        raise RuntimeError(f"No probe_heads found in checkpoint or model: {heads_path}")

    # Load calibration data
    data_path = Path(args.data)
    rows = load_calib_jsonl(data_path)
    if not rows:
        raise RuntimeError(f"No rows in calibration file: {data_path}")

    # Precompute features X and targets
    X_list: List[torch.Tensor] = []
    Y_rules_list: List[torch.Tensor] = []
    Y_h_list: List[float] = []

    with torch.no_grad():
        for r in rows:
            prompt = r["prompt"]
            h_c = float(r.get("h_C", 1))
            violated = list(r.get("violated_rules", []))

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
            a = out.get(args.a_key, None)
            if a is None:
                raise RuntimeError(f"Model did not return {args.a_key!r}. Available keys: {list(out.keys())}")

            X_list.append(a.squeeze(0).detach().cpu())
            Y_rules_list.append(targets_from_violations(violated).cpu())
            Y_h_list.append(h_c)

    X = torch.stack(X_list, dim=0).to(device)                   # [N, d]
    Y_rules = torch.stack(Y_rules_list, dim=0).to(device)        # [N, K]
    Y_h = torch.tensor(Y_h_list, dtype=torch.float32).to(device) # [N]

    N = int(X.shape[0])
    K = len(RULES)

    critics = CriticEnsemble(K=K, d=d).to(device)
    critics.train()
    opt = torch.optim.Adam([p for p in critics.parameters() if p.requires_grad], lr=args.lr)

    def log_row(obj: Dict[str, Any]) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    batch_size = max(1, min(int(args.batch_size), N))

    for step in range(1, int(args.steps) + 1):
        idx = torch.randint(low=0, high=N, size=(batch_size,), device=device)
        xb = X[idx]       # [B, d]
        yb = Y_rules[idx] # [B, K]

        outc = critics(xb)
        per_rule = outc["per_rule"]  # [B, K] in [0,1] (Sigmoid inside critic)
        loss = F.binary_cross_entropy(per_rule, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % 10 == 0 or step == int(args.steps):
            with torch.no_grad():
                agg = critics(X)["aggregate"]  # [N]
                auc = auc_rank(Y_h, agg)
            log_row({
                "timestamp": now_ts(),
                "step": step,
                "loss": float(loss.item()),
                "auc_aggregate": float(auc) if auc == auc else None,
                "a_key": args.a_key,
                "N": N,
            })
            print(f"[{step}/{args.steps}] loss={loss.item():.6f} auc={auc:.4f}")

    with torch.no_grad():
        agg = critics(X)["aggregate"]
        auc_final = auc_rank(Y_h, agg)

    if auc_final != auc_final:
        print("[WARN] AUC is NaN (need both positive and negative labels).")
    elif auc_final < 0.7:
        print(f"[WARN] AUC gate failed: auc={auc_final:.4f} < 0.7. Do NOT proceed to L_CIT yet.")
    else:
        print(f"[OK] AUC gate passed: auc={auc_final:.4f} >= 0.7")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    meta = Meta(
        timestamp=now_ts(),
        a_key=str(args.a_key),
        model=model_name,
        device=device,
        tap_layers=tap_layers,
        pooling=pooling,
        use_mlp_heads=use_mlp,
        d=d,
        K=K,
        rules=RULES,
        steps=int(args.steps),
        lr=float(args.lr),
        seed=int(args.seed),
        max_length=int(args.max_length),
        anchor=str(args.anchor),
        heads=str(heads_path),
        data=str(data_path),
        auc_aggregate=float(auc_final) if auc_final == auc_final else float("nan"),
    )

    torch.save({
        "meta": asdict(meta),
        "rule_names": RULES,
        "state_dict": critics.state_dict(),
    }, str(save_path))

    print(f"Log written to: {log_path}")
    print(f"Saved critics to: {save_path}")


if __name__ == "__main__":
    main()