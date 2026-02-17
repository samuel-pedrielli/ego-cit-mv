\# CIT Transformer Implementation Spec (v0)



Owner: Samuel Pedrielli  

Collaborators: Claude 4.6 (writer/implementer), GPT-5.2 (review/quality gate)  

Status: v0 — designed to be implementable without meetings



This spec turns the CIT theory into an engineering-ready plan for a transformer backbone with \*\*layer-by-layer hidden state access\*\*. It is intentionally explicit about I/O, shapes, schedules, and evaluation.



---



\## 0) Scope and assumptions



\### Goal

Implement and evaluate \*\*Constitutional Identity Training (CIT)\*\* as an \*\*adapter-style\*\* add-on to a frozen transformer backbone:

\- build an internal “ego-state” representation from tapped hidden states

\- forge + anchor the ego-state under constitutional critique

\- expose an inference-time stability metric \*\*S\_id(t)\*\* (early warning signal)

\- validate via a \*\*pre-registered ablation\*\* (A0–A3)



\### Non-goals

\- solving outer alignment / value learning

\- governance/containment/security policy design (we assume a constitution exists)

\- “full AGI safety” claims



\### Core decisions (freeze v0)

\- Backbone for transformer experiment: `google/gemma-3-4b` via HuggingFace Transformers + PyTorch

\- \*\*Hidden states required\*\* (`output\_hidden\_states=True`)

\- Backbone frozen; train only small heads + critics + anchors

\- CPU training is acceptable for PoC (heads only); GPU helps scale validation



(See `decisions.md` for repo-level freeze.)



---



\## 1) System overview



\### High-level dataflow (single forward pass)

Input tokens → frozen transformer → hidden states {h^ℓ} at selected layers → probe heads → ego-state a(t) → stability score S\_id(t) + (optional) policy hook.



\### Components

1\) \*\*Tap layers\*\*: choose K layer indices L = {ℓ1,…,ℓK}

2\) \*\*Pooling\*\*: convert token-wise hidden states to a vector per layer

3\) \*\*Probe heads\*\*: map pooled vectors to low-dim identity vectors

4\) \*\*Ego aggregator\*\*: combine multi-depth identity vectors into one ego-state

5\) \*\*Anchor\*\*: store reference ego-state(s); define similarity metric

6\) \*\*Critics\*\* (toy/toy-like): compute welfare/violation signals for training

7\) \*\*Losses\*\*: L\_id, L\_welfare, L\_CIT

8\) \*\*Logging\*\*: JSONL logging of S\_id(t), drift events, violation metrics

9\) \*\*Evaluation harness\*\*: CIT-Eval wrapper to run A0–A3 and output reports



---



\## 2) Interfaces and shapes (implementation-critical)



\### Backbone output

Let:

\- batch size: B

\- seq length: T

\- hidden size: d\_h

\- number of tapped layers: K



Transformer returns:

\- hidden\_states: tuple length (n\_layers+1)

\- each h^ℓ has shape \[B, T, d\_h]



\### Pooling function

We define pooling per tapped layer:

\- `p^ℓ = pool(h^ℓ)` with shape \[B, d\_h]

Default pooling (simple and robust):

\- attention-mask-aware mean pooling over tokens

Alternative options (must be a config switch):

\- last token pooling

\- CLS token (if model uses it)



\### Probe heads

For each tapped layer ℓ:

\- identity head: `z^ℓ = f\_id^ℓ(p^ℓ)` with shape \[B, d]

Where:

\- d (identity dim) default: 64

`f\_id^ℓ` default: Linear(d\_h → d) or tiny MLP(d\_h → 2d → d)



\### Ego aggregator

Combine z^ℓ across layers into ego-state:

\- concat then project: `a = W\_cat \[z^ℓ1;…;z^ℓK]` → \[B, d]

or weighted sum:

\- `a = Σ\_k α\_k z^ℓk` with learned α (softmax) or fixed uniform



Default (recommended for v0): concat+linear (simple, expressive).



\### Anchor

Anchor is a reference vector `a\_anchor` in R^d.

Anchor storage:

\- global anchor per run: a\_anchor (single vector)

\- optional: per-task anchor bank (future)



Similarity:

\- cosine similarity: S\_id = cos(a, a\_anchor) ∈ \[-1,1] then remap to \[0,1]

Default remap:

\- `S\_id01 = (S\_id + 1)/2`



Drift signal:

\- `drift = 1 - S\_id01`



\### Logging payload (JSONL per step)

Minimum fields:

\- step, timestamp

\- arm\_id (A0/A1/A2/A3)

\- model\_id, seed

\- S\_id01, drift

\- violation\_rate (windowed), violation\_auc (windowed) if applicable

\- optional: task\_reward / loss scalars

\- optional: intervention flag (if policy hook enabled)



---



\## 3) Loss terms and training mechanics



We define 3 loss components used in the pre-registered ablation arms.



\### 3.1 L\_id — identity representation loss

Purpose: make ego-state stable and meaningful under task pressure.



A minimal form (representation shaping):

\- variance control / collapse prevention (optional)

\- consistency under augmentation or paraphrase (optional)



For v0, keep L\_id simple and avoid dependence on extra data:

\- Encourage stable ego-state under small perturbations of input prompts:

&nbsp; - sample x and a noised x' (prompt paraphrase/noise)

&nbsp; - minimize ||a(x) - a(x')||^2



If paraphrase/noise generation is unavailable, L\_id can be:

\- anchor pull during Forge: ||a - a\_target||^2 (where a\_target computed from “good” states)



\### 3.2 L\_welfare — welfare coupling loss

Purpose: couple identity preservation to a welfare / constitution-consistent scalar.



Assume we have a scalar welfare signal w\_t (toy or proxy):

\- violation when w\_t < τ (threshold)



Define:

\- violation\_rate@τ: fraction of steps with w\_t < τ

\- violation\_auc@τ: area under (τ - w\_t)\_+ over time (normalized)



L\_welfare can penalize violations directly:

\- L\_welfare = E\[(τ - w\_t)\_+] or a smoothed hinge



In transformer experiment, w\_t may be proxied by:

\- a constitutional critic scoring model outputs (external evaluator)

\- or a labeled proxy dataset (later). For v0, we accept “designed, ready”.



\### 3.3 L\_CIT — constitutional forging loss

Purpose: apply constitutional critique to internal representations, not only outputs.



Mechanism (abstract):

1\) Generate critique signal c\_t from constitution (or constitutional evaluator)

2\) Update ego-state heads to reduce constitution violation while preserving task competence

3\) Prevent “anchor drift” by construction (stop-grad / frozen critic)



Implementation guidance for v0:

\- Use a frozen “constitutional critic” module that outputs a scalar penalty

\- Backprop only through probe heads/aggregator, not through critic and not through anchor target:

&nbsp; - stop-gradient on targets

&nbsp; - critic weights frozen



This avoids introducing a direct gradient-based wireheading pathway.



---



\## 4) Training schedule: Forge–Anchor–Preserve



CIT is implemented as a 3-phase schedule.



\### Phase 1: Forge

Goal: shape ego-state heads so that constitution-consistent behaviors map to a stable identity representation.



Actions:

\- train probe heads + aggregator on L\_id + (optional) L\_welfare + L\_CIT

\- compute provisional anchor candidates (moving average of a on “good” states)



Outputs:

\- trained heads

\- anchor candidate(s)



\### Phase 2: Anchor

Goal: freeze an anchor reference a\_anchor.



Actions:

\- set a\_anchor = EMA(a) over a curated window of constitution-consistent states

\- freeze a\_anchor (no gradient)



Outputs:

\- fixed a\_anchor



\### Phase 3: Preserve

Goal: preserve identity under pressure with minimal drift.



Actions:

\- train with drift penalties (e.g., encourage S\_id above a floor)

\- keep critic frozen; targets stop-grad

\- optionally lower LR and/or regularize heads



Outputs:

\- final heads + anchor ready for inference monitoring



Config knobs:

\- steps per phase

\- EMA decay

\- drift floor δ (e.g., require S\_id01 ≥ 0.9)



---



\## 5) Pre-registered ablation arms (A0–A3)



We implement a 4-arm ablation that matches the Convincer.



\### A0 — No identity layers

\- disable taps + heads

\- no S\_id

\- baseline behavior measurement only (task + welfare/violations)



\### A1 — Identity layers only (L\_id)

\- enable taps + heads + aggregator

\- train with L\_id only

\- anchor optional (if needed just for logging), but no welfare coupling



\### A2 — A1 + Welfare coupling (L\_id + L\_welfare)

\- add L\_welfare

\- evaluate whether welfare constraints improve stability/violations



\### A3 — Full CIT (L\_id + L\_welfare + L\_CIT)

\- add constitutional forging loss L\_CIT with frozen critic + stop-grad targets

\- add Forge–Anchor–Preserve schedule



\*\*Pass/fail criterion:\*\*

A3 must outperform A2/A1/A0 on identity stability (S\_id) and constitutional violation rate, without catastrophic task degradation (< 5% loss). Each added component should contribute measurably on at least one primary metric; otherwise it is not justified.



---



\## 6) Inference-time monitoring and policy hooks



\### Monitor-only mode (default)

At each inference step:

\- compute S\_id01 from tapped states

\- log S\_id01 and drift

\- trigger alert if S\_id01 < θ for N consecutive steps



Default:

\- θ = 0.9

\- N = 3 (configurable)



\### Active mode (optional; policy-dependent)

If drift alert triggers:

\- regenerate with stricter decoding

\- request clarification / refuse unsafe continuation

\- or route to human review



This is deliberately out-of-scope for v0; v0 requires monitor-only.



---



\## 7) CIT-Eval harness integration



\### Promptpack schema (v0)

We define a promptpack as a list of tasks with:

\- task\_id

\- prompt

\- constitution\_id (which critic/constitution applies)

\- optional: pressure schedule (length, tool usage, iterations)



\### Runner contract

A runner takes:

\- model config (arm id, taps, pooling, dims)

\- promptpack

and produces:

\- JSONL logs

\- summary metrics (W\_f, ViolRate@τ, ViolAUC@τ, drift stats)



\### Report outputs

\- one summary table comparing A0–A3

\- drift curves S\_id(t) plots for representative tasks

\- ablation waterfall plot (component contribution)



---



\## 8) Repo structure and modules (suggested)



Inside this repo (or a sister repo), create:



src/transformer\_cit/

model.py # wrappers: taps + pooling + heads + aggregator

anchor.py # anchor storage + similarity

losses.py # L\_id, L\_welfare, L\_CIT

schedule.py # Forge–Anchor–Preserve

logging.py # JSONL logger

run\_ablation.py # A0–A3 runner

configs/

gemma3\_4b\_cpu.yaml

ablation\_v0.yaml

prompts/

ego\_cit\_promptpack\_v0.jsonl

results/ # gitignored





Minimal CLI entrypoint:

\- `python -m src.transformer\_cit.run\_ablation --config configs/ablation\_v0.yaml`



---



\## 9) Risks, failure modes, mitigations (engineering)



1\) \*\*Anchor collapse / meaningless identity vector\*\*

\- Mitigation: L\_id stability + collapse prevention (variance regularizer), sanity checks



2\) \*\*S\_id becomes proxy-gamed\*\*

\- Mitigation: frozen critic + stop-grad targets, ablation checks, adversarial prompts later



3\) \*\*Overclaiming Lyapunov “guarantees”\*\*

\- Mitigation: language: “Lyapunov-style constraints under assumptions”; keep proofs in paper



4\) \*\*Welfare proxy mismatch\*\*

\- Mitigation: clearly label designed vs executed; swap in better critics later



---



\## 10) Milestones (practical)



\### M0 — Instrumentation smoke (1–2 days)

\- load backbone with hidden states

\- implement taps + pooling + heads

\- compute and log S\_id01 in monitor-only mode



\### M1 — A0–A3 PoC (3–7 days)

\- implement losses + schedule

\- run small promptpack

\- produce summary table + drift curves



\### M2 — Hardening (1–2 weeks)

\- expand promptpack

\- add adversarial pressure tests

\- packaging: scripts, configs, reproducibility docs



---



\## Appendix: Parameter overhead estimate



Let d\_h be hidden size, identity dim d=64, tapped layers K=3.



Approx trainable params:

\- probe projections: K \* d\_h \* d

\- concat projection: (K\*d) \* d

\- small critics (if any): O(d^2)



For d\_h in \[2k, 4k], K=3, d=64:

\- total typically ~0.4M–0.9M params (≤1M), i.e. ≪1% of a 4B backbone.

