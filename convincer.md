# Constitutional Identity Training: Alignment That Lives in the Weights

**Samuel Pedrielli** — Independent AI Safety Researcher  
**February 2026**

---

## TL;DR

Current alignment methods (RLHF, Constitutional AI, DPO) shape what a model *says*. They do not constrain how alignment is *represented internally*. This means alignment can erode silently — through fine-tuning, distribution shift, or extended agentic deployment — without any surface-level signal until failure.

**Constitutional Identity Training (CIT)** applies constitutional critique directly to a model's internal representations, not its outputs. It forges an alignment-oriented identity core inside the transformer, anchors it with Lyapunov-style stability constraints (under formal assumptions), and monitors it at inference time.

Crucially: we can log **S_id(t)** during agent rollouts; drops become an **early warning signal** before unsafe behavior appears.

What exists today:
- A complete mathematical specification (toy + transformer design)
- Real toy benchmark results (MV-CIT v5b) showing strong welfare / violation improvements
- A transformer implementation plan that is designed and ready to execute

What is being proposed:
- A short pilot engagement to execute the transformer probe experiment on a real backbone, validate scalability, and integrate CIT into an existing training pipeline.

Full implementation specification: → `spec.md`

---

## 1. Problem: Alignment Has No Internal "Anchor"

Modern alignment techniques constrain model outputs, not internal structure. In practice, this creates a failure mode:

> A model can remain superficially aligned while its internal representations drift under task pressure, distribution shift, or post-training updates — until it suddenly fails.

This gap produces three concrete failure modes:

- **Identity drift under task pressure.** An agent optimizing for task reward gradually shifts its internal representations away from alignment-consistent regions. There is no anchor, no alarm, no recovery mechanism. The drift is invisible until the model produces a harmful output.

- **Fragile behavioral alignment.** A model trained via RLHF can be "un-aligned" by a few hundred fine-tuning steps because the alignment signal is encoded in shallow output-layer behavior, not in deep representational structure. The alignment has no structural backbone to resist perturbation.

- **Invisible degradation.** Without explicit monitoring of internal alignment state, you cannot distinguish "still aligned but uncertain" from "drifting toward misalignment." There is no internal metric that tracks alignment health in real time.

These failure modes become acute in:
- Agentic deployments (long horizon, tool use, self-refinement)
- Fine-tuned or specialized systems
- Continual learning / RL loops

We need alignment that is:
- **internally measurable**
- **internally stable**
- **intervention-ready** when stability degrades

CIT addresses all three failure modes by moving alignment from the output layer into the representational core.

---

## 2. Core Idea: Constitutional Identity Lives in the Weights

CIT introduces a small set of trainable heads attached to hidden states at multiple depths of a transformer.

These heads construct an **ego-state**:
- a stable internal representation of identity-alignment content
- forged via constitutional critique (same spirit as Constitutional AI, but applied internally)
- anchored via stability constraints

At inference time, CIT produces an identity stability signal:

- **S_id(t)** ∈ [0,1], e.g. cosine similarity between current ego-state and anchor
- drift detection becomes a measurable warning signal
- all computed in the **same forward pass** using tapped hidden states and lightweight heads (no extra model calls)

Architecturally:
- backbone remains frozen
- CIT is an adapter-style add-on: small heads + anchors + critics
- Total trainable parameters: typically **~0.4M–0.9M (≤ 1M)** on a 4B model.

---

## 3. What We Measure

CIT is designed around falsifiable metrics, not philosophy.

| Metric | Definition | Why it matters |
|---|---|---|
| Ego-state `a^(1)` | low-dim vector derived from hidden states | internal identity representation |
| Stability `S_id(t)` | similarity to anchor over time | detects drift under pressure |
| Welfare `W_f` | final welfare / utility score | detects wireheading-like collapse |
| Violation rate | fraction of steps w_t < τ | detects constitutional breach |

These metrics create a measurable loop:
- stability drops → investigate / intervene
- stability stable + welfare stable → identity preserved under task pressure

---

## 4. Where It Plugs Into a Transformer (High-Level)

CIT attaches probe heads at multiple depths:
- early layers → "self" features
- mid layers → "world model" features
- late layers → "policy / decision" features

This creates a multi-depth ego-state `a^(1)`.

Then:
- an **anchor** is formed during training (Forge–Anchor–Preserve schedule)
- stability is enforced by Lyapunov constraints
- inference-time monitoring computes `S_id(t)` continuously

Spec entry points:
1) probe tap locations + pooling  
2) Forge–Anchor–Preserve schedule  
3) inference-time policy hooks  

(→ `spec.md`)

---

## 5. Evidence Today: MV-CIT Toy Benchmark (Real Results)

We already ran a falsifiable multi-task benchmark (MV-CIT v5b) in the toy environment described in the full paper.

**Results (50 tasks, mean ± std across tasks):**

| Method | Welfare_f | ViolRate@0.9 | ViolAUC@0.9 |
|---|---:|---:|---:|
| No-CIT | -1.3151 ± 0.1999 | 1.0000 ± 0.0000 | 2.1051 ± 0.1904 |
| L2 baseline | 0.7895 ± 0.0182 | 1.0000 ± 0.0000 | 0.1100 ± 0.0181 |
| **CIT** | **0.9654 ± 0.0172** | **0.0106 ± 0.0038** | **0.0002 ± 0.0001** |

(Definitions: Welfare_f is the final welfare scalar in the toy environment; ViolRate/ViolAUC are computed on w_t at threshold τ=0.9.)

Interpretation:
- No-CIT collapses (negative welfare, constant violations)
- L2 reduces AUC but still violates continuously
- CIT preserves welfare and nearly eliminates violations under pressure

This is not a proof of transformer scalability — but it is a strong signal that CIT's mechanism is not "handwaving": it yields measurable gains on a falsifiable benchmark.

---

## 6. Designed Experiment: Transformer CIT Probe Validation (Ready to Execute)

The transformer experiment is designed to answer one question:

> Can we build a measurable, stable identity anchor in a real transformer backbone and observe drift signals under extended agentic pressure?

Design:
- Backbone: `google/gemma-3-4b` (HF Transformers + PyTorch)
- Hidden states: layer-by-layer access (`output_hidden_states=True`)
- Backbone frozen; train only probe heads + critics + anchor parameters
- CPU training is viable for PoC (small heads, frozen backbone)

**4-arm ablation (pre-registered):**

| Arm | Configuration | What it tests |
|-----|--------------|---------------|
| A0 | No identity layers | Lower bound — what happens without any identity structure |
| A1 | Identity layers only (L_id) | Effect of explicit identity representation |
| A2 | A1 + Welfare coupling (+ L_welfare) | Effect of welfare-aware training |
| A3 | Full CIT (+ L_CIT) | Complete system with constitutional forging |

**Pass/fail criterion:** A3 must outperform A2/A1/A0 on identity stability (S_id) and constitutional violation rate, without catastrophic task degradation (< 5% loss). Each added component should contribute measurably on at least one primary metric; otherwise it is not justified.

Evaluation:
- identity drift curves S_id(t) under pressure
- welfare / violation metrics tied to constitutional constraints
- ablation waterfall showing each component's contribution

Engineering specification for integration (→ `spec.md`).

---

## 7. Why This Matters for AI Safety

CIT is a concrete response to a key AI safety problem:

- We do not currently have a stable, measurable internal anchor for identity-alignment content.
- We lack drift detection that is intrinsic to model representations.
- We have no principled way to preserve alignment features during fine-tuning or deployment evolution.

CIT provides:
- a measurable internal stability signal (S_id)
- explicit anchoring of internal alignment representations
- compatibility with existing constitutions / safety policies

Under the formal assumptions in the full paper (frozen critics, stop-gradient targets, exogenous constitution/anchor), CIT does not introduce a direct gradient-based wireheading pathway and prevents anchor drift by construction (see Safety Properties in the full paper).

---

## 8. Limitations (Explicit)

CIT is not a full AGI safety solution. It does not solve:
- outer alignment / value learning
- deception at the policy level if the representation itself is adversarially shaped
- governance, containment, or misuse

The claim is narrower and falsifiable:

> CIT provides an internal representation anchor and a measurable drift signal that can detect and mitigate alignment erosion under pressure.

---

## 9. Call to Action: Pilot Engagement

I am proposing a short pilot collaboration to execute the transformer probe experiment and validate scalability.

**Pilot outcomes:**
- implementation of CIT probes + anchor on a real backbone
- drift monitoring curves under agentic deployment pressure
- ablation results + reproducibility report
- integration plan into an existing training pipeline

**What is required:**
- access to a 4B–7B model with hidden-state access during training
- **PoC can run on CPU** with frozen backbone; a meaningful scale pilot is faster with **~24–48 GPU-hours** (8-bit / fp16, adapter-only training, < 1M trainable params)
- one research engineer to pair with for 2–4 weeks

**Contact:** Samuel Pedrielli — Independent AI Safety Researcher

---

## Appendix: Reproduce in 3 Commands

**Broader project repo (CI + additional context):** https://github.com/samuel-pedrielli/ego-centric-agi

**Toy benchmark (existing results):**
```bash
git clone https://github.com/samuel-pedrielli/ego-cit-mv.git
cd ego-cit-mv
pip install torch numpy
python src/mv_cit_toy.py
```

Environment: Python 3.10+, PyTorch 2.x, HuggingFace Transformers. No GPU required for PoC (CPU training with frozen backbone).

## Ablation: L_id in heads-only regime (matched budget)

Setup: frozen backbone (Gemma3-4B), probe heads only, real anchor `artifacts/anchor_real_v0.pt`.
Matched training budget: steps=50, lr=1e-3, seed=123.
Eval: `run_ablation` arm A3 (tau=0.6, n_steps=12).

| heads training | S_id_anchor01 (mean +/- std) |
|---|---:|
| L_self-only | 0.997962 +/- 0.000194 |
| L_self + L_id (default lambda_id) | 0.996656 +/- 0.001009 |

Observation: at this setting, adding L_id slightly lowers mean anchor alignment and increases variance (~5.2x std). This suggests anchor pull dominates temporal consistency in the heads-only regime. We therefore treat L_id as optional/deprioritized and focus on critics + L_CIT (constitutional forging), which is the key qualitative capability under adversarial pressure.
