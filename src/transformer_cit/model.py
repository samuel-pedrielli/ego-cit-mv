"""CIT Model: frozen backbone + concentric probe heads (v2).

Changes from v1:
- ProbeHead: optional LayerNorm for scale-invariant gradients.
  With LayerNorm ON, gradient magnitude is independent of backbone
  hidden_size — critical for multi-scale (4B → 72B) experiments.
- forward(): optional return_pooled for efficient preserve computation
  without re-running the backbone.

Backward-compatible: old configs without use_layernorm still work.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class ProbeHead(nn.Module):
    """Projects pooled hidden state [B, d_h] -> identity vector [B, d].

    When use_layernorm=True (default), hidden states are normalized
    before projection.  This ensures that the gradient w.r.t. probe
    weights has the same magnitude regardless of whether d_h = 2048
    (Gemma-4B) or 8192 (Qwen-72B).
    """

    def __init__(self, d_h: int, d: int, use_mlp: bool = False,
                 use_layernorm: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        if use_layernorm:
            layers.append(nn.LayerNorm(d_h))
        if use_mlp:
            layers.extend([
                nn.Linear(d_h, 2 * d),
                nn.GELU(),
                nn.Linear(2 * d, d),
            ])
        else:
            layers.append(nn.Linear(d_h, d))
        self.proj = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class CITModel(nn.Module):
    """Wraps a frozen backbone with concentric identity probes.

    tap_layers: 0-indexed layer indices.
        Gemma-3-4b  (34 layers): [11, 22, 33]
        Qwen2.5-72B (80 layers): [25, 52, 77]
    """

    def __init__(self, model_name: str, tap_layers: list[int],
                 d: int = 64, pooling: str = "mean",
                 use_mlp_heads: bool = False,
                 use_layernorm: bool = True,
                 quantize_4bit: bool = False,
                 torch_dtype: str = "float32"):
        super().__init__()

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        model_dtype = dtype_map.get(torch_dtype, torch.float32)

        load_kwargs: dict = {
            "output_hidden_states": True,
            "torch_dtype": model_dtype,
        }

        if quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=model_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["quantization_config"] = bnb_config
                load_kwargs["device_map"] = "auto"
                del load_kwargs["torch_dtype"]
                print(f"[INFO] Loading with 4-bit quantization (nf4)")
            except ImportError:
                print("[WARN] bitsandbytes not available, loading without quantization")

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tap_layers = tap_layers
        self.pooling = pooling

        # --- Discover hidden size ---
        cfg = self.backbone.config
        d_h = getattr(cfg, "hidden_size", None)
        if d_h is None and hasattr(cfg, "text_config"):
            d_h = getattr(cfg.text_config, "hidden_size", None)
        if d_h is None:
            d_h = int(self.backbone.get_input_embeddings().weight.shape[1])
        self.hidden_size = d_h

        print(f"[INFO] Backbone: {model_name}, hidden_size={d_h}, "
              f"layers={getattr(cfg, 'num_hidden_layers', '?')}, "
              f"taps={tap_layers}, identity_dim={d}, "
              f"layernorm={'ON' if use_layernorm else 'OFF'}")

        # Probe heads always in float32 for gradient precision
        self.probe_heads = nn.ModuleList([
            ProbeHead(d_h, d, use_mlp=use_mlp_heads, use_layernorm=use_layernorm)
            for _ in tap_layers
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_pooled: bool = False,
    ) -> dict | tuple:
        """Returns concentric identity vectors {a1, a2, a3}.

        If return_pooled=True, also returns pooled backbone hidden
        states {h1, h2, h3}.  This allows computing a_nat (preserve
        reference) by applying reference probe heads to the SAME
        hidden states, without re-running the backbone forward pass.
        """
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError(
                "Backbone did not return hidden_states. "
                "Ensure output_hidden_states=True."
            )

        identity: dict[str, torch.Tensor] = {}
        pooled_dict: dict[str, torch.Tensor] = {}

        for i, (layer_idx, head) in enumerate(
            zip(self.tap_layers, self.probe_heads)
        ):
            h = hidden_states[layer_idx + 1]           # [B, T, d_h]
            pooled = self.pool(h, attention_mask).float()  # [B, d_h]
            pooled_dict[f"h{i+1}"] = pooled
            identity[f"a{i+1}"] = head(pooled)          # [B, d]

        if return_pooled:
            return identity, pooled_dict
        return identity
