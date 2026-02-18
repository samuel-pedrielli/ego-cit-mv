"""Anchor computation (Phase 2) and S_id evaluation.

Changes from previous version:
  - Added load_from_file() classmethod to restore anchor from .pt artifact
  - Confirmed: revision uses a1_rev (not a1) for gradient step
"""
import torch
import torch.nn as nn


class AnchorStore(nn.Module):
    """Stores frozen anchor mu_align and computes S_id."""
    def __init__(self, d: int):
        super().__init__()
        self.register_buffer("mu_align", torch.zeros(d))
        self.frozen = False
        self._meta = {}

    @property
    def meta(self) -> dict:
        """Read-only metadata loaded from the anchor artifact."""
        return self._meta

    @classmethod
    def load_from_file(cls, path: str) -> "AnchorStore":
        """Load anchor from .pt file produced by anchor_offline.py."""
        payload = torch.load(path, weights_only=False)
        mu = payload["mu_align"]
        store = cls(d=mu.shape[0])
        store.mu_align.copy_(mu)
        store.frozen = True
        store._meta = payload.get("meta", {})
        return store

    def compute_anchor(self, model, critics, dataloader,
                       revision_epsilon: float = 0.01):
        """
        Phase 2 (recommended: run FIRST, before Forge).
        anchor = mean of revised frozen-base representations.
        """
        model.eval()
        critics.eval()

        revised_sum = torch.zeros_like(self.mu_align)
        count = 0

        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            identity = model(input_ids, attention_mask)
            a1 = identity["a1"].detach()

            a1_rev = a1.clone().requires_grad_(True)
            result = critics(a1_rev)
            s_R = result["aggregate"]
            grad = torch.autograd.grad(s_R.sum(), a1_rev)[0]
            a1_revised = a1_rev + revision_epsilon * grad

            revised_sum += a1_revised.sum(dim=0).detach()
            count += a1_revised.size(0)

        self.mu_align.copy_(revised_sum / count)
        self.frozen = True

    def s_id(self, a1: torch.Tensor) -> torch.Tensor:
        """Cosine similarity remapped to [0, 1]. a1: [B, d]."""
        cos = nn.functional.cosine_similarity(
            a1, self.mu_align.unsqueeze(0), dim=-1
        )
        return (cos + 1.0) / 2.0

    def drift(self, a1: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.s_id(a1)
