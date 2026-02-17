"""Constitutional critics C_k: frozen classifiers on a^(1)."""
import torch
import torch.nn as nn


class ConstitutionalCritic(nn.Module):
    """Single rule critic: a^(1) [B, d] -> score [B] in [0, 1]."""
    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, a1: torch.Tensor) -> torch.Tensor:
        return self.net(a1).squeeze(-1)  # [B]


class CriticEnsemble(nn.Module):
    """K critics with per-rule weights.

    IMPORTANT: call .freeze() after training critics (Stage 0)
    and before any identity training (Stage 1+).
    """
    def __init__(self, K: int, d: int, weights: list[float] | None = None,
                 frozen: bool = False):
        super().__init__()
        self.critics = nn.ModuleList([ConstitutionalCritic(d) for _ in range(K)])
        if weights is None:
            weights = [1.0 / K] * K
        self.register_buffer("weights", torch.tensor(weights))
        if frozen:
            self.freeze()

    def forward(self, a1: torch.Tensor) -> dict:
        scores = [c(a1) for c in self.critics]              # list of [B]
        scores_t = torch.stack(scores, dim=-1)               # [B, K]
        s_R = (scores_t * self.weights).sum(dim=-1)          # [B]
        return {"per_rule": scores_t, "aggregate": s_R}

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
