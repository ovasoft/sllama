"""
Pretrained model base class for the SLLama family.

This module defines `SLLamaPreTrainedModel`, which extends
`LlamaPreTrainedModel` from Hugging Face Transformers. It adds
utility methods for parameter inspection, optimizer configuration,
and FLOPS utilization estimation (MFU). The implementation builds
on ideas from Karpathy's NanoGPT.
"""

import inspect
from collections import defaultdict
from typing import Optional

import torch
from torch import nn

from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel
)

from .configuration_sllama import SLLamaConfig  # renamed from configuration_baby


class SLLamaPreTrainedModel(LlamaPreTrainedModel):
    """
    Base class for all SLLama models.

    This class extends `LlamaPreTrainedModel` and provides:
      - Parameter counting utilities.
      - FLOPS utilization estimation.
      - Custom optimizer configuration with fused AdamW support.
      - Layer grouping and weight sharing methods for model reduction.

    Attributes:
        config (SLLamaConfig): The model configuration.
    """

    config_class = SLLamaConfig

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the total number of parameters in the model.

        Args:
            non_embedding (bool, optional): If True, exclude embedding
                parameters from the count. Defaults to True.

        Returns:
            int: Total parameter count.
        """
        n_params = sum(p.numel() for p in self.parameters())

        if not non_embedding:
            return n_params

        try:
            if hasattr(self.model, "embed_tokens"):
                embed_tokens = self.model.embed_tokens
                if hasattr(embed_tokens, "weight"):
                    n_params -= embed_tokens.weight.numel()
                else:
                    n_params -= sum(p.numel() for p in embed_tokens.parameters())
        except AttributeError:
            # Fallback for models with transformer.wpe
            if hasattr(self, "transformer") and hasattr(self.transformer, "wpe"):
                n_params -= self.transformer.wpe.weight.numel()

        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate Model FLOPS Utilization (MFU) in A100 bfloat16 peak FLOPS units.

        Args:
            fwdbwd_per_iter (int): Number of forward/backward passes per iteration.
            dt (float): Time per iteration in seconds.

        Returns:
            float: Estimated MFU value (0–1).
        """
        n_params = self.get_num_params()
        cfg = self.config

        try:
            L = cfg.num_hidden_layers
            H = cfg.num_attention_heads
            Q = cfg.hidden_size // cfg.num_attention_heads
            T = cfg.max_position_embeddings
        except AttributeError:
            L = cfg.n_layer
            H = cfg.n_head
            Q = cfg.n_embd // cfg.n_head
            T = cfg.block_size

        flops_per_token = 6 * n_params + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        flops_achieved = flops_per_iter / dt  # FLOPS per second
        flops_peak = 312e12  # 312 TFLOPS for A100 (bfloat16)
        return flops_achieved / flops_peak

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """
        Configure and return an AdamW optimizer with proper parameter grouping.

        Args:
            weight_decay (float): Weight decay factor.
            learning_rate (float): Learning rate.
            betas (tuple[float, float]): Adam betas.
            device_type (str): Device type, used to enable fused optimizer on CUDA.

        Returns:
            torch.optim.Optimizer: Configured AdamW optimizer.
        """
        param_dict = {
            name: p for name, p in self.named_parameters() if p.requires_grad
        }

        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(
            f"Decayed tensors: {len(decay_params)} ({num_decay:,} params), "
            f"Non-decayed tensors: {len(nodecay_params)} ({num_nodecay:,} params)"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"Using fused AdamW: {use_fused}")
        return optimizer

    def group_layers(self) -> dict[int, list[nn.Module]]:
        """
        Group decoder layers into shared-weight groups.

        Returns:
            dict[int, list[nn.Module]]: Mapping from group index to layers.
        """
        cfg = self.config
        grouping = [
            int(i / cfg.num_hidden_layers * cfg.n_group)
            for i in range(cfg.num_hidden_layers)
        ]

        groups = defaultdict(list)
        for layer_idx, group_idx in enumerate(grouping):
            groups[group_idx].append(self.layers[layer_idx])

        return groups

    def reduce_mlp_weight(self, layers: list[nn.Module], part: Optional[str] = None):
        """
        Share up and down projection weights in MLP layers.

        Args:
            layers (list): Layers whose MLP weights should be reduced/shared.
            part (str, optional): Placeholder for interface consistency.
        """

        def share_up_and_down(layer: nn.Module):
            layer.mlp.up_proj.weight = layer.mlp.down_proj.weight

        for layer in layers:
            share_up_and_down(layer)

    def layer_share_weights(
        self, grouped_layers: dict[int, list[nn.Module]], part: Optional[str] = None
    ):
        """
        Share attention and/or MLP weights across grouped layers.

        Args:
            grouped_layers (dict): Mapping of group indices to layer lists.
            part (str, optional): 'attention', 'mlp', or 'all'.
        """

        def share_attention(first_layer: nn.Module, layer: nn.Module):
            if hasattr(layer.self_attn, "shared_weights") and hasattr(
                first_layer.self_attn, "shared_weights"
            ):
                layer.self_attn.shared_weights = first_layer.self_attn.shared_weights
            else:
                layer.self_attn.q_proj.weight = first_layer.self_attn.q_proj.weight
                layer.self_attn.k_proj.weight = first_layer.self_attn.k_proj.weight
                layer.self_attn.v_proj.weight = first_layer.self_attn.v_proj.weight
                layer.self_attn.o_proj.weight = first_layer.self_attn.o_proj.weight

        def share_mlp(first_layer: nn.Module, layer: nn.Module):
            layer.mlp.gate_proj.weight = first_layer.mlp.gate_proj.weight
            layer.mlp.up_proj.weight = first_layer.mlp.up_proj.weight
            layer.mlp.down_proj.weight = first_layer.mlp.down_proj.weight

        for _, layers in grouped_layers.items():
            if len(layers) <= 1:
                continue

            first_layer = layers[0]
            for layer in layers[1:]:
                if part == "attention":
                    share_attention(first_layer, layer)
                elif part == "mlp":
                    share_mlp(first_layer, layer)
                elif part == "all":
                    share_attention(first_layer, layer)
                    share_mlp(first_layer, layer)
