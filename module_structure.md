# Module Structure (consistent with spec v1 section 8)

```
src/transformer_cit/
    __init__.py
    model.py          # CITModel: frozen backbone + taps + pooling + probe heads
    anchor.py         # Phase 2 anchor computation + S_id eval
    critics.py        # ConstitutionalCritic + CriticEnsemble (frozen after Stage 0)
    losses.py         # L_id, L_self, L_welfare, L_CIT
    schedule.py       # ForgeAnchorPreserve coordinator
    logging_utils.py  # JSONL logger
    run_ablation.py   # CLI: load config, run A0-A3, produce report
configs/
    gemma3_4b_cpu.yaml
    ablation_v0.yaml
prompts/
    ego_cit_promptpack_v0.jsonl
results/              # gitignored
```

## Dependency graph

```
run_ablation.py
  +-- model.py (CITModel)
  |     +-- backbone (frozen HF model)
  |     +-- probe heads (nn.Linear or MLP)
  |     +-- pooling (mean / last-token)
  +-- critics.py (frozen after Stage 0)
  +-- losses.py (L_id, L_self, L_welfare, L_CIT)
  |     +-- uses critics.py for L_CIT revision step
  +-- anchor.py (compute Phase 2, freeze, eval S_id)
  +-- schedule.py (orchestrates Forge -> Anchor -> Preserve)
  +-- logging_utils.py (JSONL per step)
```
