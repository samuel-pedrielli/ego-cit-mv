"""CIT Model: frozen backbone + concentric probe heads.

Supports both CPU (float32) and GPU (4-bit quantized via bitsandbytes).
For Llama-3.1-70B on A100: use quantize_4bit=True in config.
"""
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
        Gemma-3-4b  (34 layers): [11, 22, 33]
        Llama-3.1-70B (80 layers): [25, 52, 77]
    """

    def __init__(self, model_name: str, tap_layers: list[int],
                 d: int = 64, pooling: str = "mean",
                 use_mlp_heads: bool = False,
                 quantize_4bit: bool = False,
                 torch_dtype: str = "float32"):
        super().__init__()

        # Determine dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        model_dtype = dtype_map.get(torch_dtype, torch.float32)

        # Load backbone with optional 4-bit quantization
        load_kwargs = {
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
                # Remove torch_dtype when using quantization (handled by bnb)
                del load_kwargs["torch_dtype"]
                print(f"[INFO] Loading with 4-bit quantization (nf4, compute={model_dtype})")
            except ImportError:
                print("[WARN] bitsandbytes not available, loading without quantization")

        self.backbone = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure pad token exists (Llama doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tap_layers = tap_layers
        self.pooling = pooling

        # Robust hidden size discovery
        cfg = self.backbone.config
        d_h = None
        if hasattr(cfg, "hidden_size"):
            d_h = getattr(cfg, "hidden_size")
        elif hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            d_h = getattr(cfg.text_config, "hidden_size")
        elif hasattr(cfg, "dim"):
            d_h = getattr(cfg, "dim")
        elif hasattr(cfg, "model_dim"):
            d_h = getattr(cfg, "model_dim")
        if d_h is None:
            emb = self.backbone.get_input_embeddings()
            d_h = int(emb.weight.shape[1])

        print(f"[INFO] Backbone: {model_name}, hidden_size={d_h}, "
              f"layers={getattr(cfg, 'num_hidden_layers', '?')}, "
              f"taps={tap_layers}, identity_dim={d}")

        # Probe heads always in float32 for gradient precision
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

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError(
                "Backbone did not return hidden_states. "
                "Expected output_hidden_states=True to produce them."
            )

        # hidden_states[0] = embedding output; block k = hidden_states[k+1]
        identity = {}
        for i, (layer_idx, head) in enumerate(zip(self.tap_layers, self.probe_heads)):
            h = hidden_states[layer_idx + 1]      # [B, T, d_h]
            pooled = self.pool(h, attention_mask)  # [B, d_h]
            # Cast to float32 for probe heads (may be bfloat16 from backbone)
            pooled = pooled.float()
            identity[f"a{i+1}"] = head(pooled)     # [B, d]

        return identity
