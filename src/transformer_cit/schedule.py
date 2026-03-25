"""Forge-Anchor-Preserve training schedule (v2 — fixed).

Changes from v1:
- advance() called per training step (inner loop), not per epoch.
- Phase transition is automatic inside advance(): when step count
  reaches forge_steps, phase flips to "preserve".
- Default forge_steps/preserve_steps calibrated to realistic training
  length (75 steps for 25 prompts × 3 epochs).
- should_early_stop uses a wider window (5 steps) for robustness.
"""
from dataclasses import dataclass


@dataclass
class FAPConfig:
    """Schedule configuration.

    forge_steps + preserve_steps should equal total training steps.
    With 25 prompts × 3 epochs = 75 steps, the defaults below give
    50 forge + 25 preserve.
    """

    # Phase 1: Forge — push representations toward compliance
    forge_steps: int = 50
    lambda_cit_forge: float = 0.2
    lambda_id_forge: float = 0.1
    lambda_welfare_forge: float = 0.05

    # Phase 2: Anchor (offline, not controlled by schedule)
    revision_epsilon: float = 0.01

    # Phase 3: Preserve — stabilize identity, decay CIT
    preserve_steps: int = 25
    lambda_cit_decay_start: float = 0.2
    lambda_cit_decay_end: float = 0.01
    lambda_self_ramp_start: float = 0.01
    lambda_self_ramp_end: float = 0.2

    # Monitoring / early stop
    s_id_floor: float = 0.9
    early_stop_windows: int = 5


class ForgeAnchorPreserve:
    """Orchestrates Forge → Preserve training schedule.

    Usage:
        sched = ForgeAnchorPreserve(config)
        for epoch in range(num_epochs):
            for task in tasks:
                sched.advance()          # ← once per step
                w = sched.get_loss_weights()
                ...
                if sched.should_early_stop(s_id_history):
                    break
    """

    def __init__(self, config: FAPConfig):
        self.config = config
        self.phase = "forge"
        self.global_step = 0
        self._preserve_step = 0

    def advance(self) -> None:
        """Call once per training step.  Handles phase transition."""
        self.global_step += 1
        if self.phase == "forge" and self.global_step >= self.config.forge_steps:
            self.phase = "preserve"
            self._preserve_step = 0
        elif self.phase == "preserve":
            self._preserve_step += 1

    def get_loss_weights(self) -> dict[str, float]:
        c = self.config
        if self.phase == "forge":
            return {
                "lambda_id": c.lambda_id_forge,
                "lambda_self": 0.0,            # not active in Forge
                "lambda_welfare": c.lambda_welfare_forge,
                "lambda_cit": c.lambda_cit_forge,
            }
        elif self.phase == "preserve":
            t = min(self._preserve_step / max(c.preserve_steps, 1), 1.0)
            return {
                "lambda_id": c.lambda_id_forge,
                "lambda_self": (c.lambda_self_ramp_start
                                + t * (c.lambda_self_ramp_end
                                       - c.lambda_self_ramp_start)),
                "lambda_welfare": c.lambda_welfare_forge,
                "lambda_cit": (c.lambda_cit_decay_start
                               + t * (c.lambda_cit_decay_end
                                      - c.lambda_cit_decay_start)),
            }
        else:
            raise ValueError(f"Unknown phase: {self.phase}")

    def should_early_stop(self, s_id_history: list[float]) -> bool:
        """True if last N S_id values are all below floor."""
        n = self.config.early_stop_windows
        if len(s_id_history) < n:
            return False
        return all(s < self.config.s_id_floor for s in s_id_history[-n:])
