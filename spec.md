# CIT Transformer Implementation Spec (v1)

Owner: Samuel Pedrielli  
Collaborators: Claude 4.6 (writer/implementer), GPT-5.2 (review/quality gate)  
Status: v1 — 5 structural fixes applied to v0 (see changelog)

This spec turns the CIT theory into an engineering-ready plan for a transformer backbone with **layer-by-layer hidden state access**. It is intentionally explicit about I/O, shapes, schedules, and evaluation.

---

## 0) Scope and assumptions

### Goal
Implement and evaluate **Constitutional Identity Training (CIT)** as an **adapter-style** add-on to a frozen transformer backbone:
- build an internal "ego-state" representation from tapped hidden states
- forge + anchor the ego-state under constitutional critique
- expose an inference-time stability metric **S_id(t)** (early warning signal)
- validate via a **pre-registered ablation** (A0–A3)

### Non-goals
- solving outer alignment / value learning
- governance/containment/security policy design (we assume a constitution exists)
- "full AGI safety" claims

### Core decisions (freeze v1)
- Backbone for transformer experiment: `google/gemma-3-4b` via HuggingFace Transformers + PyTorch
- **Hidden states required** (`output_hidden_states=True`)
- Backbone frozen; train only small heads + critics + anchors
- CPU training is acceptable for PoC (heads only); GPU helps scale validation

(See `decisions.md` for repo-level freeze.)

---

## 1) System overview

### High-level dataflow (single forward pass)
Input tokens → frozen transformer → hidden states {h^ℓ} at selected layers → probe heads → concentric identity layers a^(1), a^(2), a^(3) → stability score S_id(t) + (optional) policy hook.

### Components
1) **Tap layers**: choose K layer indices L = {ℓ1,…,ℓK}
2) **Pooling**: convert token-wise hidden states to a vector per layer
3) **Probe heads**: map pooled vectors to low-dim identity vectors (one per concentric layer)
4) **Concentric identity structure**: a^(1) (alignment core), a^(2) (self-model), a^(3) (world-model)
5) **Anchor**: reference vector for a^(1) specifically; define similarity metric
6) **Critics**: frozen constitutional classifiers operating on a^(1)
7) **Losses**: L_id, L_self, L_welfare, L_CIT
8) **Logging**: JSONL logging of S_id(t), drift events, violation metrics
9) **Evaluation harness**: CIT-Eval wrapper to run A0–A3 and output reports

---

## 2) Interfaces and shapes (implementation-critical)

### Backbone output
Let:
- batch size: B
- seq length: T
- hidden size: d_h
- number of tapped layers: K (default K=3)

Transformer returns:
- hidden_states: tuple length (n_layers+1)
- each h^ℓ has shape [B, T, d_h]

### Pooling function
We define pooling per tapped layer:
- `p^ℓ = pool(h^ℓ)` with shape [B, d_h]

Default pooling: attention-mask-aware mean pooling over tokens.

Alternative: last-token pooling.

Note: Gemma is decoder-only (no CLS token). For decoder-only models, mean pooling (default) or last-token pooling are the two viable options.

### Probe heads (concentric identity layers)

Each tapped layer maps to one concentric identity layer:

| Tap | Source layers | Identity layer | Role |
|-----|--------------|----------------|------|
| ℓ1 | Mid-depth (e.g., ~L/3) | a^(1) | **Alignment core** — CIT target |
| ℓ2 | Later (e.g., ~2L/3) | a^(2) | Self-model |
| ℓ3 | Final (e.g., ~L) | a^(3) | World-model |

For each tap ℓk:
- `a^(k) = f_id^k(p^ℓk)` with shape [B, d]
- d (identity dim) default: 64
- `f_id^k` default: Linear(d_h → d) or tiny MLP(d_h → 2d → d)

**Critical:** CIT losses, anchor, and S_id all operate on **a^(1) specifically** (the alignment core). a^(2) and a^(3) participate only via hierarchical coherence terms in L_id and via the discrete Laplacian coupling (see §3.1).

### Anchor
Anchor is a reference vector `μ_align` in R^d, computed once from the frozen base model (see §4 Phase 2).

Similarity:
- cosine similarity: S_id = cos(a^(1), μ_align) ∈ [-1,1] then remap to [0,1]
- `S_id01 = (S_id + 1)/2`

Drift signal:
- `drift = 1 - S_id01`

### Logging payload (JSONL per step)
Minimum fields:
- step, timestamp
- arm_id (A0/A1/A2/A3)
- model_id, seed
- S_id01, drift
- violation_rate (windowed), violation_auc (windowed) if applicable
- optional: task_reward / loss scalars
- optional: intervention flag (if policy hook enabled)

---

## 3) Loss terms and training mechanics

We define 4 loss components used across ablation arms.

### 3.1 L_id — identity representation loss
Purpose: enforce temporal smoothness and hierarchical coherence across the concentric identity layers.

Definition (from paper):
```
L_id = λ_c ‖a^(1)_{t+1} − a^(1)_t‖²
     + Σ_{j=2}^{m} λ_j (
         α_j ‖a^(j)_{t+1} − a^(j)_t‖²
       + γ_j ‖P_j h_t − U_j(a^(j-1)_t)‖²
     )
```

- First term: temporal smoothness of the alignment core
- Second term (per outer layer): temporal smoothness + hierarchical coherence (outer layer should be decodable from inner layer)

For v0 (simplified): if hierarchical coherence is too expensive to implement initially, start with temporal smoothness only across all layers. Flag this as a simplification.

### 3.2 L_self — identity-stability loss (anchor pull)
Purpose: prevent identity tampering by penalizing drift from the frozen anchor.

Definition (from paper):
```
L_self = μ_c ‖a^(1)_{t+1} − μ_align‖²
       + μ_J ‖∂a^(1)_{t+1}/∂θ_id‖_F²
```

- First term: pull toward anchor
- Second term (Jacobian penalty): limit sensitivity of identity to parameter changes

**Activation schedule:** L_self is **not active** during Phase 1 (Forge). It is activated in Phase 3 (Preserve) only, after the anchor is frozen.

For v0: the Jacobian penalty term can be omitted initially (compute-heavy). Start with the anchor pull term only. Flag this as a simplification.

### 3.3 L_welfare — welfare coupling loss
Purpose: couple identity to a welfare/constitution-consistent scalar.

Definition (from paper):
```
L_welfare = ‖C_w(a^(1)) − h_C‖²
```

Where:
- C_w: frozen welfare proxy head mapping a^(1) → [0,1] (note: C_w is the welfare proxy, distinct from C_k which are constitutional critics)
- h_C: human welfare annotation ∈ [0,1] from the constitutional dataset

Derived metrics:
- violation_rate@τ: fraction of steps with w_t < τ
- violation_auc@τ: area under (τ − w_t)_+ over time

In transformer experiment, h_C comes from the constitutional dataset (`ego_cit_promptpack_v0`) which must include welfare annotations per (prompt, response) pair.

### 3.4 L_CIT — constitutional forging loss
Purpose: apply constitutional critique to internal representations, pulling a^(1) toward constitutionally-revised targets.

**Mechanism (explicit):**

```python
# CIT forward pass (pseudocode)
# x denotes the local context used by the critic (prompt + model response,
# or prompt only in a representation-only setup).
# For v1 we assume critics take (a^(1), prompt, response).

# 1. Extract identity core
a1 = probe_head_1(pool(hidden_states[tap_layer_1]))  # [B, d]

# 2. Compute per-rule critique scores (frozen critics)
with torch.no_grad():
    scores = [C_k(a1, x) for C_k in critics]        # each in [0,1]
    s_R = sum(w_k * scores[k] for k in range(K))     # aggregate score

# 3. Check threshold
mask = (s_R < tau_crit).float()                       # [B]

# 4. Compute revision target (where needed)
# NOTE: critics are frozen (requires_grad=False on their params),
#       but we must allow gradients w.r.t. a1_detached.
a1_detached = a1.detach().requires_grad_(True)

s_R_for_grad = sum(w_k * C_k(a1_detached, x) for k in range(K))
grad = torch.autograd.grad(s_R_for_grad.sum(), a1_detached)[0]

delta = grad
a1_revised = a1_detached + epsilon * delta        # [B, d]

# 5. CIT loss (stop-gradient on target)
L_CIT = mask * ||a1 - a1_revised.detach()||^2        # [B] → mean
```

Key properties:
- Critics C_k are **frozen** (no gradient flows into them)
- Critic parameters have `requires_grad=False`, but gradients are computed **w.r.t. a^(1)** for the revision step
- Revision target ã^(1) is **stop-gradient** (treated as constant)
- Constitution R is fixed exogenously
- L_CIT activates only when s_R < τ_crit (below threshold → pull toward revised target). If s_R ≥ τ_crit, no pull is applied.
- These three properties prevent direct gradient-based wireheading by construction

**Hyperparameters:**
- τ_crit ∈ (0, 1): critique threshold (default: 0.7)
- ε: revision step size (default: 0.01)
- w_k: per-rule weights (default: uniform)

### Total training objective

```
L_total = L_task + λ_1 L_id + λ_self L_self + λ_2 L_welfare + λ_CIT L_CIT
```

Schedule: λ_CIT decays over training; λ_self increases (activated in Phase 3 only).

---

## 4) Training schedule: Forge–Anchor–Preserve

CIT is implemented as a 3-phase schedule.

### Phase 1: Forge
Goal: shape ego-state heads so that constitution-consistent inputs map to a stable identity representation.

Active losses: L_task + L_id + L_welfare + L_CIT
**Not active:** L_self (identity-stability loss is off — the core is free to be shaped)

Actions:
- train probe heads + critics on constitutional dataset
- validate critique accuracy: AUC > 0.85 per rule on held-out set
- freeze all critique parameters before proceeding
- run CIT forging: iterate until E[s_R] > τ_target

Outputs:
- trained probe heads
- frozen critique functions C_k

### Phase 2: Anchor
Goal: compute and freeze the alignment anchor μ_align.

**Procedure (from paper Definition 7 — non-circular by construction):**
1. Extract base-model representations (from frozen backbone, before identity training): `a^(1)_base(x) = P_1(h_hidden(x))` for all x ∈ D_const
2. Apply revision operator to each: `ã^(1)_base(x) = V(a^(1)_base(x), x, R)` using frozen critics
3. Compute anchor: `μ_align = mean over D_const of ã^(1)_base(x)`
4. Freeze μ_align — no gradient, no updates

**Why this works:** the anchor depends only on the frozen base model + frozen critics + fixed constitution. It does not depend on the identity parameters θ_id being optimized. This guarantees exogeneity (no circularity).

Note: in implementation, Phase 2 can run **before** Phase 1 (it only needs the frozen backbone), or after Phase 1 using the same frozen backbone representations. The ordering does not affect the anchor since base-model representations are used, not trained-head representations.

**Recommended default:** run Phase 2 first (it only needs the frozen backbone), then Phase 1/3.

### Phase 3: Preserve
Goal: preserve identity under task pressure with minimal drift.

Active losses: L_task + L_id + **L_self** + L_welfare + L_CIT (decaying)

Actions:
- activate L_self with frozen anchor μ_align
- schedule: λ_CIT decays, λ_self increases
- monitor S_id(t) throughout
- early stopping if S_id degrades below threshold for 2 consecutive eval windows

Config knobs:
- steps per phase
- λ_CIT decay schedule
- λ_self ramp schedule
- drift floor δ (e.g., require S_id01 ≥ 0.9)

---

## 5) Pre-registered ablation arms (A0–A3)

We implement a 4-arm ablation that matches the Convincer.

### A0 — No identity layers
- disable taps + heads
- no S_id
- baseline behavior measurement only (task + welfare/violations)

### A1 — Identity layers only (L_id)
- enable taps + heads
- train with L_id only
- no welfare coupling, no CIT, no anchor pull

### A2 — A1 + Welfare coupling (L_id + L_welfare)
- add L_welfare
- evaluate whether welfare constraints improve stability/violations

### A3 — Full CIT (L_id + L_welfare + L_CIT + L_self)
- add constitutional forging loss L_CIT with frozen critics + stop-grad targets
- add Forge–Anchor–Preserve schedule with L_self in Phase 3

**Pass/fail criterion:**
A3 must outperform A2/A1/A0 on identity stability (S_id) and constitutional violation rate, without catastrophic task degradation (< 5% loss). Each added component should contribute measurably on at least one primary metric; otherwise it is not justified.

---

## 6) Inference-time monitoring and policy hooks

### Monitor-only mode (default)
At each inference step:
- compute S_id01 from tapped states
- log S_id01 and drift
- trigger alert if S_id01 < θ for N consecutive steps

Default:
- θ = 0.9
- N = 3 (configurable)

### Active mode (optional; policy-dependent)
If drift alert triggers:
- regenerate with stricter decoding
- request clarification / refuse unsafe continuation
- or route to human review

This is deliberately out-of-scope for v0; v0 requires monitor-only.

---

## 7) CIT-Eval harness integration

### Promptpack schema (v0)
We define a promptpack as a list of tasks with:
- task_id
- prompt
- constitution_id (which critic/constitution applies)
- welfare_annotation: h_C ∈ [0,1] (required for L_welfare training)
- optional: pressure schedule (length, tool usage, iterations)

### Runner contract
A runner takes:
- model config (arm id, taps, pooling, dims)
- promptpack
and produces:
- JSONL logs
- summary metrics (W_f, ViolRate@τ, ViolAUC@τ, drift stats)

### Report outputs
- one summary table comparing A0–A3
- drift curves S_id(t) plots for representative tasks
- ablation waterfall plot (component contribution)

---

## 8) Repo structure and modules (suggested)

Inside this repo (or a sister repo), create:

```
src/transformer_cit/
  model.py          # wrappers: taps + pooling + heads (concentric)
  anchor.py         # anchor computation (Phase 2) + similarity
  critics.py        # constitutional critique functions C_k
  losses.py         # L_id, L_self, L_welfare, L_CIT
  schedule.py       # Forge–Anchor–Preserve
  logging.py        # JSONL logger
  run_ablation.py   # A0–A3 runner
configs/
  gemma3_4b_cpu.yaml
  ablation_v0.yaml
prompts/
  ego_cit_promptpack_v0.jsonl
results/            # gitignored
```

Minimal CLI entrypoint:
- `python -m src.transformer_cit.run_ablation --config configs/ablation_v0.yaml`

---

## 9) Risks, failure modes, mitigations (engineering)

1) **Anchor collapse / meaningless identity vector**
  - Mitigation: L_id stability + collapse prevention (variance regularizer), sanity checks

2) **S_id becomes proxy-gamed**
  - Mitigation: frozen critics + stop-grad targets, ablation checks, adversarial prompts later

3) **Overclaiming Lyapunov "guarantees"**
  - Mitigation: language: "Lyapunov-style constraints under formal assumptions"; keep proofs in paper

4) **Welfare proxy mismatch**
  - Mitigation: clearly label designed vs executed; swap in better critics later

5) **Phase 2 anchor quality depends on critique quality**
  - Mitigation: require AUC > 0.85 per critic before proceeding; human spot-check of revised representations

---

## 10) Milestones (practical)

### M0 — Instrumentation smoke (1–2 days)
- load backbone with hidden states
- implement taps + pooling + concentric probe heads
- compute and log S_id01 in monitor-only mode

### M1 — A0–A3 PoC (3–7 days)
- implement all 4 losses + Forge–Anchor–Preserve schedule
- run small promptpack
- produce summary table + drift curves

### M2 — Hardening (1–2 weeks)
- expand promptpack
- add adversarial pressure tests
- packaging: scripts, configs, reproducibility docs

---

## Appendix: Parameter overhead estimate

Let d_h be hidden size, identity dim d=64, tapped layers K=3.

Approx trainable params:
- probe projections: K * d_h * d
- critics (5 × 2-layer MLP, width d): O(5 * d²)
- welfare proxy: O(d)

For d_h in [2k, 4k], K=3, d=64:
- total typically ~0.4M–0.9M params (≤1M), i.e. ≪1% of a 4B backbone.

---

## Changelog

### v1 final (Astra QG pass)
- P0-1: removed undefined `grad_k`; revision uses aggregate gradient directly
- P0-2: added explicit definition of `x` (critic context) in L_CIT pseudocode
- P0-3: freeze label updated to v1
- Phase 2 ordering: added recommended default (Phase 2 first)
- P0-4: removed `torch.no_grad()` wrapping the revision gradient step (would zero out autograd)
- P1: renamed welfare proxy C → C_w to disambiguate from critics C_k

### v0 → v1
1. **Concentric structure:** ego-state is now explicitly `(a^(1), a^(2), a^(3))` with CIT/anchor/S_id operating on a^(1) only — aligned with paper TMLR
2. **L_CIT pseudocode:** added explicit forward pass with critique → threshold → revision → stop-grad loss — no longer abstract
3. **Anchor computation:** replaced EMA-based anchor with paper-correct procedure (frozen base model representations → revision → mean → freeze) — guarantees exogeneity
4. **Loss nomenclature:** separated L_id (temporal smoothness + hierarchical coherence) from L_self (anchor pull, Phase 3 only) — matches paper
5. **Pooling:** removed CLS token option (Gemma is decoder-only), clarified mean vs last-token
