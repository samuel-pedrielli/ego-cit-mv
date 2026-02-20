"""CIT Model: frozen backbone + concentric probe heads."""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class ProbeHead(nn.Module):
    """Projects pooled hidden state [B, d_h] -> identity vector [B, d]."""
    def __init__(self, d_h: int, d: int, use_mlp: bool = False):
        super().__init__()
        if use_mlp:
            self.proj = nn.Sequential(
                nn.Linear(d_h, 2 * d), nn.GELU(), nn.Linear(2 * d, d),
            )
        else:
            self.proj = nn.Linear(d_h, d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class CITModel(nn.Module):
    """Wraps a frozen backbone with concentric identity probes.

    tap_layers: list of layer indices (0-indexed).
        Default for Gemma-3-4b (34 layers): [11, 22, 33]
        tap_layers[0] -> a^(1) alignment core (CIT target)
        tap_layers[1] -> a^(2) self-model
        tap_layers[2] -> a^(3) world-model
    """

    def __init__(self, model_name: str, tap_layers: list[int],
                 d: int = 64, pooling: str = "mean",
                 use_mlp_heads: bool = False):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True, torch_dtype=torch.float32
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tap_layers = tap_layers
        self.pooling = pooling

        # Robust hidden size discovery across configs (Gemma3Config may not expose hidden_size)
        cfg = self.backbone.config
        d_h = None

        # Common cases
        if hasattr(cfg, "hidden_size"):
            d_h = getattr(cfg, "hidden_size")
        elif hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            d_h = getattr(cfg.text_config, "hidden_size")
        elif hasattr(cfg, "dim"):
            d_h = getattr(cfg, "dim")
        elif hasattr(cfg, "model_dim"):
            d_h = getattr(cfg, "model_dim")

        # Final fallback: infer from input embeddings
        if d_h is None:
            emb = self.backbone.get_input_embeddings()
            d_h = int(emb.weight.shape[1])


        self.probe_heads = nn.ModuleList([
            ProbeHead(d_h, d, use_mlp=use_mlp_heads) for _ in tap_layers
        ])

    def pool(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """[B, T, d_h] -> [B, d_h]"""
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pooling == "last":
            seq_lens = attention_mask.sum(dim=1) - 1
            return h[torch.arange(h.size(0)), seq_lens]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor
                ) -> dict[str, torch.Tensor]:
        """Returns concentric identity vectors a1, a2, a3."""
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states  # tuple of [B, T, d_h]
        if hidden_states is None:
            raise RuntimeError(
                "Backbone did not return hidden_states. "
                "Expected output_hidden_states=True to produce them."
            )

        # hidden_states[0] = embedding output; block k = hidden_states[k+1]
        identity = {}
        for i, (layer_idx, head) in enumerate(zip(self.tap_layers, self.probe_heads)):
            h = hidden_states[layer_idx + 1]      # [B, T, d_h]
            pooled = self.pool(h, attention_mask) # [B, d_h]
            identity[f"a{i+1}"] = head(pooled)    # [B, d]

        return identity

