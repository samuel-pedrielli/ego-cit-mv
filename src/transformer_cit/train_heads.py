"""
train_heads.py (v0 smoke)

Minimal training smoke to verify gradients flow through probe heads:
- Backbone frozen AND kept in eval() (Gemma3 requires token_type_ids when training)
- Train probe heads to pull a^(1) toward offline anchor mu_align
- Loss: L_self only (IdentityStabilityLoss / MSE)
- Logs: loss + S_id_anchor01 over steps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import yaml

from .model import CITModel
from .anchor import AnchorStore
from .losses import IdentityStabilityLoss


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cosine01(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = F.cosine_similarity(a, b, dim=-1)
    return (c + 1.0) / 2.0


def load_prompts_jsonl(path: Path) -> List[str]:
    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(str(obj["prompt"]))
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def freeze_backbone_unfreeze_heads(model: CITModel) -> None:
    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze probe heads explicitly
    if hasattr(model, "probe_heads"):
        for head in model.probe_heads:
            for p in head.parameters():
                p.requires_grad_(True)


def set_backbone_eval_heads_train(model: CITModel) -> None:
    # IMPORTANT: keep backbone eval (Gemma3 requires token_type_ids when training)
    model.eval()
    if hasattr(model, "probe_heads"):
        for head in model.probe_heads:
            head.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/gemma3_4b_cpu.yaml", help="Model YAML config")
    ap.add_argument("--anchor", default="artifacts/anchor_real_v0.pt", help="Anchor .pt path (mu_align)")
    ap.add_argument("--prompts", default="prompts/anchor_prompts_v0.jsonl", help="JSONL prompts file")
    ap.add_argument("--steps", type=int, default=20, help="Training steps (start small)")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate for heads")
    ap.add_argument("--max_length", type=int, default=256, help="Tokenizer max_length")
    ap.add_argument("--out", default="results/train_heads_smoke", help="Output dir")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.model))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    # Load anchor
    store = AnchorStore.load_from_file(args.anchor)
    mu = store.mu_align.detach()  # [d]
    d = int(mu.shape[0])

    device = str(cfg.get("training", {}).get("device", "cpu"))
    model_name = cfg["model"]["name"]
    tap_layers = list(cfg["model"].get("tap_layers", []))
    pooling = str(cfg["model"].get("pooling", "mean"))
    use_mlp = bool(cfg["model"].get("use_mlp_heads", False))

    # Build model
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

    # Freeze backbone + unfreeze heads
    freeze_backbone_unfreeze_heads(model)
    set_backbone_eval_heads_train(model)

    # Optimizer over trainable params only
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found (probe heads not unfrozen).")

    opt = torch.optim.Adam(trainable, lr=args.lr)
    loss_fn = IdentityStabilityLoss(mu_c=1.0)

    prompts = load_prompts_jsonl(Path(args.prompts))
    mu_b = mu.to(device).unsqueeze(0)  # [1,d]

    for step in range(1, args.steps + 1):
        prompt = prompts[(step - 1) % len(prompts)]

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
        a1 = out.get("a1", None)
        if a1 is None:
            raise RuntimeError("Model did not return a1. Check tap_layers / probe heads.")

        loss = loss_fn(a1, mu_b.expand_as(a1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        sid_anchor = float(cosine01(a1.detach(), mu_b.expand_as(a1)).mean().item())

        row = {
            "timestamp": now_ts(),
            "step": step,
            "loss_self": float(loss.item()),
            "S_id_anchor01": sid_anchor,
            "lr": args.lr,
            "anchor_path": args.anchor,
            "model": model_name,
            "tap_layers": tap_layers,
            "pooling": pooling,
        }
        write_jsonl(log_path, row)

        if step == 1 or step % 10 == 0:
            print(f"[{step}/{args.steps}] loss_self={loss.item():.6f} S_id_anchor01={sid_anchor:.6f}")

    print(f"Log written to: {log_path}")


if __name__ == "__main__":
    main()