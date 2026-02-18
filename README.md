# ego-cit-mv

Minimal, reproducible toy experiments for MV-CIT (Critique-and-Iterative-Transformation), plus the outreach docs:
- `convincer.md` (Convincer Kit)
- `spec.md` (Transformer implementation spec)

This repo is meant to be easy to run locally and easy to review.

## Contents
- `src/mv_cit_toy.py` — standalone script reproducing the multi-task toy benchmark (v5b).
- `notebooks/mv_cit_toy.ipynb` — notebook used to generate results.
- `results/` — outputs (CSV/TXT summaries).  
- `convincer.md` — short outreach document (AI safety framing + experiment plan).
- `spec.md` — implementation-ready transformer spec (A0–A3 ablation, losses, schedule).

## Quick start (Windows, local)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch numpy
python src\mv_cit_toy.py
Outputs are written under results/.

Quick start (Linux/macOS, local)
python -m venv .venv
source .venv/bin/activate
pip install torch numpy
python src/mv_cit_toy.py
Colab (optional)
Open notebooks/mv_cit_toy.ipynb in Google Colab (upload it, or open it from GitHub).

Notes
This is a toy benchmark repo. The transformer experiment is specified in spec.md and is designed to be implemented on a frozen backbone with hidden-state access.

## Transformer CIT skeleton (dry smoke)

Install deps (Windows):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Dry run (no model download):

python -m src.transformer_cit.run_ablation --dry --arms A0 --out results/dry_smoke
Expected outputs:

results/dry_smoke/A0/a0_log.jsonl

results/dry_smoke/A0/a0_summary.txt