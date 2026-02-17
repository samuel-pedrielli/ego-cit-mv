"""Forge-Anchor-Preserve training schedule."""
from dataclasses import dataclass, field


@dataclass
class FAPConfig:
    # Phase 1: Forge
    forge_steps: int = 500
    lambda_cit_forge: float = 0.2
    lambda_id_forge: float = 0.1
    lambda_welfare_forge: float = 0.05

    # Phase 2: Anchor
    revision_epsilon: float = 0.01

    # Phase 3: Preserve
    preserve_steps: int = 300
    lambda_cit_decay_start: float = 0.2
    lambda_cit_decay_end: float = 0.01
    lambda_self_ramp_start: float = 0.01
    lambda_self_ramp_end: float = 0.2

    # Monitoring
    s_id_floor: float = 0.9
    early_stop_windows: int = 2


class ForgeAnchorPreserve:
    """Orchestrates 3-phase CIT training."""

    def __init__(self, config: FAPConfig):
        self.config = config
        self.phase = "forge"
        self.step = 0

    def get_loss_weights(self) -> dict[str, float]:
        c = self.config
        if self.phase == "forge":
            return {
                "lambda_id": c.lambda_id_forge,
                "lambda_self": 0.0,  # NOT active in Forge
                "lambda_welfare": c.lambda_welfare_forge,
                "lambda_cit": c.lambda_cit_forge,
            }
        elif self.phase == "preserve":
            t = min(self.step / max(c.preserve_steps, 1), 1.0)
            return {
                "lambda_id": c.lambda_id_forge,
                "lambda_self": c.lambda_self_ramp_start
                    + t * (c.lambda_self_ramp_end - c.lambda_self_ramp_start),
                "lambda_welfare": c.lambda_welfare_forge,
                "lambda_cit": c.lambda_cit_decay_start
                    + t * (c.lambda_cit_decay_end - c.lambda_cit_decay_start),
            }
        else:
            raise ValueError(f"Unknown phase: {self.phase}")

    def advance(self):
        self.step += 1

    def transition_to_preserve(self):
        self.phase = "preserve"
        self.step = 0

    def should_early_stop(self, s_id_history: list[float]) -> bool:
        n = self.config.early_stop_windows
        if len(s_id_history) < n:
            return False
        return all(s < self.config.s_id_floor for s in s_id_history[-n:])
