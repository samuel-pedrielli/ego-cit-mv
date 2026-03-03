from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

# Ensure repo root is on sys.path so we can import src.transformer_cit.*
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.transformer_cit.model import CITModel  # noqa: E402
from src.transformer_cit.anchor import AnchorStore  # noqa: E402

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("PyYAML is required (yaml). Please install it in your venv.") from e


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Meta:
    timestamp: str
    note: str
    seed: int
    model: str
    device: str
    tap_layers: List[int]
    pooling: str
    use_mlp_heads: bool
    d: int
    anchor: str
    out: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/gemma3_4b_cpu.yaml", help="Model YAML config")
    ap.add_argument("--anchor", required=True, help="Anchor .pt path (mu_align) to infer d")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for head init")
    ap.add_argument("--out", default="artifacts/heads_init_seed123.pt", help="Output checkpoint path")
    args = ap.parse_args()

    # seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = load_yaml(Path(args.model))
    device = str(cfg.get("training", {}).get("device", "cpu"))

    model_name = cfg["model"]["name"]
    tap_layers = list(cfg["model"].get("tap_layers", []))
    pooling = str(cfg["model"].get("pooling", "mean"))
    use_mlp = bool(cfg["model"].get("use_mlp_heads", False))

    # discover d from anchor
    store = AnchorStore.load_from_file(args.anchor)
    d = int(store.mu_align.shape[0])

    # build model (this initializes probe heads)
    model = CITModel(
        model_name=model_name,
        tap_layers=tap_layers,
        d=d,
        pooling=pooling,
        use_mlp_heads=use_mlp,
    ).to(device)

    # extract probe heads state dicts in the same format expected by run_ablation.py
    if not hasattr(model, "probe_heads"):
        raise RuntimeError("CITModel has no attribute probe_heads")
    probe_heads = []
    for h in model.probe_heads:
        probe_heads.append(h.state_dict())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = Meta(
        timestamp=now_ts(),
        note="init-only probe heads (no training); for critics-first diagnostics",
        seed=int(args.seed),
        model=model_name,
        device=device,
        tap_layers=tap_layers,
        pooling=pooling,
        use_mlp_heads=use_mlp,
        d=d,
        anchor=str(args.anchor),
        out=str(out_path),
    )

    torch.save(
        {
            "meta": asdict(meta),
            "probe_heads": probe_heads,
        },
        str(out_path),
    )

    print(f"Saved init heads to: {out_path}")
    print(f"meta: {asdict(meta)}")


if __name__ == "__main__":
    main()