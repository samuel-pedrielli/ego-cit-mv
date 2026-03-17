"""CIT loss terms: L_id, L_self, L_welfare, L_CIT (direct optimization)."""
import torch
import torch.nn as nn


class IdentityLoss(nn.Module):
    """L_id: temporal smoothness of concentric identity layers."""
    def __init__(self, lambda_c: float = 0.1):
        super().__init__()
        self.lambda_c = lambda_c

    def forward(self, a_curr: dict, a_prev: dict) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(iter(a_curr.values())).device)
        for key in a_curr:
            if key in a_prev:
                loss = loss + self.lambda_c * (
                    a_curr[key] - a_prev[key]
                ).pow(2).mean()
        return loss


class IdentityStabilityLoss(nn.Module):
    """L_self: anchor pull. Active only in Phase 3 (Preserve)."""
    def __init__(self, mu_c: float = 0.2):
        super().__init__()
        self.mu_c = mu_c

    def forward(self, a1: torch.Tensor, mu_align: torch.Tensor) -> torch.Tensor:
        return self.mu_c * (a1 - mu_align).pow(2).mean()


class WelfareLoss(nn.Module):
    """L_welfare: ||C_w(a^(1)) - h_C||^2."""
    def forward(self, c_w_output: torch.Tensor, h_C: torch.Tensor) -> torch.Tensor:
        return (c_w_output - h_C).pow(2).mean()


class CITLoss(nn.Module):
    """L_CIT: constitutional forging loss (direct optimization).

    Directly optimizes critic score for non-compliant samples.
    Activates only when s_R < tau_crit.
    Critics must be frozen (requires_grad=False on their params).

    Gradient flow: loss -> critic_score -> critics(a_crit) -> a_crit -> probe_heads
    Critics parameters are NOT updated (frozen); only probe heads receive gradients.
    """
    def __init__(self, tau_crit: float = 0.7, epsilon: float = 0.01):
        super().__init__()
        self.tau_crit = tau_crit
        self.epsilon = epsilon  # kept for config compatibility, not used

    def forward(self, a1: torch.Tensor, critics: nn.Module) -> torch.Tensor:
        """a1: [B, d] with grad. critics: frozen CriticEnsemble."""
        # 1. Compute critic scores (grad flows through a1 -> critics -> score)
        s_R = critics(a1)["aggregate"]  # [B]

        # 2. Threshold mask (no grad needed for mask)
        with torch.no_grad():
            mask = (s_R < self.tau_crit).float()  # [B]
        if mask.sum() == 0:
            return torch.tensor(0.0, device=a1.device)

        # 3. Direct loss: minimize negative critic score for non-compliant samples
        #    loss = -mean(score[non_compliant]) -> gradient pushes score UP
        return -(mask * s_R).sum() / mask.sum()
