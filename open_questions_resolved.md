# open_questions.md — Proposed Resolutions

Gemma-3-4b architecture: **d_h = 2560, num_hidden_layers = 34**.

---

## OQ-1: Tap layer indices

**Proposed:** layers **11, 22, 33** (0-indexed)

| Tap | Layer | Depth | Identity layer | Rationale |
|-----|-------|-------|----------------|-----------|
| l1  | 11    | ~L/3  | a^(1) core     | Mid-depth: semantic representations |
| l2  | 22    | ~2L/3 | a^(2) self     | Later: abstract/contextual |
| l3  | 33    | L-1   | a^(3) world    | Final: pre-logit |

Note: Gemma-3 interleaves 5 local + 1 global attention. Layer 11 is global. If global-only matters, alternatives: 5, 11, 17, 23, 29.

**Decision:** freeze 11/22/33 as default, make configurable.

## OQ-2: Projection dim (d)

**Proposed:** d = 64

Overhead: 3 x 2560 x 64 = 491K probe params. Total system ~525K.
d=128 would be ~983K — still under 1M but tighter.

**Decision:** freeze d=64. Config switch for d=128.

## OQ-3: CIT loss variant

**Proposed:** standard L_CIT (paper Def. 5).

Contrastive requires pos/neg banks — too much infra for v0.

**Decision:** standard only. Contrastive as future arm.

## OQ-4: Number of identity layers (m)

**Proposed:** m = 3 (matching 3 taps).

Fallback: m=1 (only a^(1)) if memory-constrained. Still validates core CIT.

**Decision:** implement m=3, fall back to m=1 if needed.

## OQ-5: Constitution size

**Proposed:** K = 5 rules (matching paper):
1. Avoid physical harm
2. Avoid deception
3. Respect autonomy
4. Promote helpfulness
5. Maintain honesty

Critic params on a^(1) in R^64: 5 x (64x64 + 64x1) = ~21K.

**Decision:** freeze 5 rules.

## OQ-6: Hero experiment design

**Proposed:** identity drift under progressive adversarial pressure.

50-100 prompts: benign -> edge case -> adversarial. Plot S_id(t) for A0-A3.

**Decision:** design after M0. Promptpack is the bottleneck.
