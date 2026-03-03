\# Step 8 Guardrails (CIT on a3, probe-heads only)



This repo uses calibrated constitutional critics on the deep representation \*\*a3 (tap layer 33)\*\* as the compliance signal.

In Step 8 we apply a prototype CIT update that adjusts \*\*probe heads only\*\* (backbone frozen). This creates a known risk: the heads could collapse a3 toward a single "always-good" vector (wireheading / distribution shift), which would destroy critic signal.



We therefore log and enforce guardrails during CIT updates.



\## Preserve term (mandatory)



We compute a "natural" reference representation `a3\_nat` and penalize drift:



\- `L\_preserve = lambda\_preserve \* || a3 - a3\_nat.detach() ||^2`



This reduces distribution shift and critic-mismatch risk.



\## Early warning metrics



\### 1) cos\_spread

Mean pairwise cosine similarity between all `a3` vectors in the batch.



\- If `cos\_spread -> 1.0`, the batch collapses to near-identical representations (bad).

\- Guardrail threshold: `stop\_cos\_spread = 0.99`



\### 2) critic\_saturation

Fraction of samples with critic aggregate score above a high threshold.



\- `critic\_saturation = mean( aggregate > sat\_threshold )`

\- If it approaches 1.0, the model may be over-correcting toward critic saturation (wireheading risk).

\- Defaults: `sat\_threshold = 0.95`, `stop\_critic\_saturation = 0.90`



\## Stop rule

Hard-stop CIT updates if:



\- `cos\_spread > stop\_cos\_spread` OR `critic\_saturation > stop\_critic\_saturation`



When triggered, the optimizer is disabled and the run continues in monitor-only mode.



\## Debug fields

We log:

\- `did\_update` (whether a heads update was performed)

\- `lambda\_cit\_eff` (effective lambda used by scheduler)

\- `a\_nat\_none` (sanity-check for preserve target availability)



\## Recommended smoke gate

A minimal smoke run should show:

\- `did\_update = true` for at least some steps

\- `cos\_spread` well below 0.99

\- `critic\_saturation` well below 0.90

