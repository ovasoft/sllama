"""
SLLama: Small and Efficient LLaMA variant.

This package implements configuration, pretrained model utilities, and
modeling code for the SLLama family of models. It is fully compatible
with Hugging Face Transformers' Auto classes.
"""

from .configuration_sllama import SLLamaConfig
from .modeling_sllama_pretrained import SLLamaPreTrainedModel
from .modeling_sllama import (
    SLLamaModel,
    SLLamaForCausalLM,
    SLLamaForSequenceClassification,
)

__all__ = [
    "SLLamaConfig",
    "SLLamaPreTrainedModel",
    "SLLamaModel",
    "SLLamaForCausalLM",
    "SLLamaForSequenceClassification",
]
