## Anchor Offline (Phase 2)

Computes the alignment anchor `mu_align` from frozen backbone + frozen critics, following spec v1 §4 Phase 2.

### Dry mode (no model download)

Tests the full pipeline with random tensors. No GPU, no download, runs in seconds:

```bash
python -m src.transformer_cit.anchor_offline --dry --out artifacts/anchor_v0.pt
```

### Real mode (CPU, downloads backbone)

Downloads `google/gemma-3-4b-it` (~8 GB), runs forward pass on the prompt pack, computes revised anchor:

```bash
pip install torch transformers pyyaml
python -m src.transformer_cit.anchor_offline \
    --model configs/gemma3_4b_cpu.yaml \
    --prompts prompts/anchor_prompts_v0.jsonl \
    --out artifacts/anchor_v0.pt
```

### Output format

The `.pt` file contains:
- `mu_align`: tensor `[d]` (default d=64) — the frozen alignment anchor
- `meta`: dict with model name, tap layers, pooling, n_samples, timestamp, git commit

### Loading the anchor

```python
from src.transformer_cit.anchor import AnchorStore

store = AnchorStore.load_from_file("artifacts/anchor_v0.pt")
print(store.mu_align.shape)  # [64]
print(store.frozen)           # True

# Compute S_id on a new a^(1):
s_id = store.s_id(a1)        # [B] in [0, 1]
```

### Notes

- `artifacts/` is gitignored — anchor output files are not committed.
- Critics are randomly initialized and frozen; they will be replaced with trained critics in M1.
- The anchor procedure is exogenous by construction: it depends only on frozen backbone + frozen critics + fixed constitution, not on trainable identity parameters.
