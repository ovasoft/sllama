"""
Configuration class for the SLLama model.

This class extends the `LlamaConfig` from Hugging Face Transformers and adds
custom parameters for model reduction and grouping strategies used in the
SLLama project.
"""

from transformers import LlamaConfig


class SLLamaConfig(LlamaConfig):
    model_type = "sllama"

    def __init__(
        self,
        n_group: int = 1,
        emb_reduction_type: str  = "repeat",
        layer_reduction_type: str  = "share",
        mlp_reduction: bool = True,
        attn_reduction_type: str = "pwa",
        hidden_size: int = 64,
        num_hidden_layers: int = 6,
        max_position_embeddings = 256,
        tie_word_embeddings: bool = False,
        intermediate_size = 64*3,
        num_key_value_heads = 4,
        num_attention_heads = 4,
        **kwargs,
    ):
        """
        Initialize the SLLama configuration.

        Args:
            n_group (int, optional): Number of layer groups for model reduction.
            emb_reduction_type (str, optional): Type of embedding reduction strategy - linear | repeat | attn
            layer_reduction_type (str, optional): Type of layer reduction strategy - share | reuse
            mlp_reduction (bool, optional): Whether to reduce MLP dimensions.
            attn_reduction_type (str, optional): Type of attention reduction strategy. - mha | skqa | pwa | rra
            **kwargs: Additional keyword arguments passed to LlamaConfig.

    
        """
        super().__init__(**kwargs)

        self.emb_reduction_type = emb_reduction_type 
        self.layer_reduction_type = layer_reduction_type 
        self.mlp_reduction = mlp_reduction
        self.attn_reduction_type = attn_reduction_type
        self.hidden_size = hidden_size or self.hidden_size
        self.num_hidden_layers = num_hidden_layers or self.num_hidden_layers
        self.n_group = n_group or self.num_hidden_layers
        self.max_position_embeddings = max_position_embeddings or self.max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings or self.tie_word_embeddings
        self.intermediate_size = intermediate_size or self.intermediate_size
        self.num_attention_heads = num_attention_heads or self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads or self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        
        
