"""
Modeling code for the SLLama model family.

This module defines the main architecture and task heads (Causal LM and
Sequence Classification) for SLLama — a small, efficient variant of LLaMA.
It includes support for embedding reduction, layer sharing, and
attention-/MLP-reduction techniques for low-resource adaptation.
"""

from collections import defaultdict
from itertools import permutations
from torch import nn
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
    LlamaForSequenceClassification,
    LlamaForCausalLM,
    LlamaAttention,
    LlamaModel,
    repeat_kv,
    apply_rotary_pos_emb,
    ACT2FN,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import DynamicCache, Cache

from .configuration_sllama import SLLamaConfig
from .modeling_sllama_pretrained import SLLamaPreTrainedModel

# ---------------------------------------------------------------------------
# Embedding-reduction modules
# ---------------------------------------------------------------------------


class LinearProjector(nn.Module):
    def __init__(self, hidden_size, proj_factor, hidden_act=None):
        super().__init__()
        self.projector = nn.Linear(hidden_size, proj_factor * hidden_size)

    def forward(self, x):
        return self.projector(x)


class AttentionProjector(nn.Module):
    def __init__(self, hidden_size, proj_factor, hidden_act=None):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, proj_factor * hidden_size)
        self.act = ACT2FN[hidden_act]

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.act(y)


class RepeatProjector(nn.Module):
    def __init__(self, hidden_size=None, proj_factor=None, hidden_act=None):
        super().__init__()
        self.proj_factor = proj_factor

    def forward(self, x):
        return x.repeat(1, 1, self.proj_factor)


class NoneProjector(nn.Module):
    def __init__(self, hidden_size=None, proj_factor=None, hidden_act=None):
        super().__init__()

    def forward(self, x):
        return x


PROJECTORS = {
    "linear": LinearProjector,
    "attn": AttentionProjector,
    "repeat": RepeatProjector,
}

# ---------------------------------------------------------------------------
# Attention-reduction modules
# ---------------------------------------------------------------------------

class SLLamaAttention(LlamaAttention):
    """Custom attention module for SLLama."""

    _tied_weights_keys = []

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        
        bsz, q_len, _ = hidden_states.size()
        #print('#'*10,query_states.shape,key_states.shape,value_states.shape)
        #raise NotImplementedError("Debugging SLLamaAttention - ignore this error")
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        #print('*'*10,query_states.shape,key_states.shape,value_states.shape)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = causal_mask is None and q_len > 1
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return attn_output, None, past_key_value


class SLLamaMHAttention(SLLamaAttention):
    """Standard multi-head attention wrapper."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        attn_output, attn_weight, past_key_value = super().forward(
            hidden_states,
            q,
            k,
            v,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            **kwargs,
        )
        return self.o_proj(attn_output), attn_weight, past_key_value

class SLLamaSKQAttention(SLLamaAttention):
    """Shared key/query attention."""

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.q_proj.weight = self.k_proj.weight
        self._tied_weights_keys.append("k_proj_weight.weight")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        attn_output, attn_weight, past_key_value = super().forward(
            hidden_states,
            q,
            k,
            v,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            **kwargs,
        )
        return self.o_proj(attn_output), attn_weight, past_key_value


class SLLamaPWAttention(SLLamaAttention):
    """Permutation-weight attention with shared embeddings."""

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.shared_weights = nn.Embedding(6, config.hidden_size)
        permutes = [i for x in permutations(range(6), 4) for i in x]
        hs = config.hidden_size
        self.q_idx = torch.tensor(permutes[:hs], dtype=torch.int)
        self.k_idx = torch.tensor(permutes[hs : hs * 2], dtype=torch.int)
        self.v_idx = torch.tensor(permutes[hs * 2 : hs * 3], dtype=torch.int)
        self.o_idx = torch.tensor(permutes[hs * 3 : hs * 4], dtype=torch.int)
        self.q_proj = self.k_proj = self.v_proj = self.o_proj = None
        if config.layer_reduction_type == "share" and layer_idx != 0:
            self._tied_weights_keys.append("shared_weights.weight")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        q = F.linear(hidden_states, self.shared_weights.weight[self.q_idx])
        k = F.linear(hidden_states, self.shared_weights.weight[self.k_idx])
        v = F.linear(hidden_states, self.shared_weights.weight[self.v_idx])
        return super().forward(
            hidden_states,
            q,
            k,
            v,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            **kwargs,
        )


class SLLamaRRWAttention(SLLamaAttention):
    """Reduced-rank weight attention with repeats."""

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        h = config.hidden_size // 4
        self.q_proj = nn.Linear(config.hidden_size, h)
        self.k_proj = nn.Linear(config.hidden_size, h)
        self.v_proj = nn.Linear(config.hidden_size, h)
        self.o_proj = nn.Linear(config.hidden_size, h)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        q = self.q_proj(hidden_states).repeat(1, 1, 4)
        k = self.k_proj(hidden_states).repeat(1, 1, 4)
        v = self.v_proj(hidden_states).repeat(1, 1, 4)
        attn_output, attn_weight, past_key_value = super().forward(
            hidden_states,
            q,
            k,
            v,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            **kwargs,
        )
        return self.o_proj(attn_output).repeat(1, 1, 4), attn_weight, past_key_value
        

EXP_ATTENTION_CLASSES = {
    "mha": SLLamaMHAttention,
    "skqa": SLLamaSKQAttention,
    "pwa": SLLamaPWAttention,
    "rra": SLLamaRRWAttention,
} 



# ---------------------------------------------------------------------------
# MLP and decoder layers
# ---------------------------------------------------------------------------


class SLLamaMLP(LlamaMLP):
    def forward(self, x):
        if x.shape[-1] == self.up_proj.weight.shape[-1]:
            up = self.up_proj(x)
        else:
            up = F.linear(x, self.up_proj.weight.T, self.up_proj.bias)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * up)


class SLLamaDecoderLayer(LlamaDecoderLayer):
    _tied_weights_keys = []

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.config = config
        attn_type = config.attn_reduction_type or "mha"
        self.self_attn = EXP_ATTENTION_CLASSES[attn_type](config, layer_idx)
        self.mlp = SLLamaMLP(config)
        self._tied_weights_keys += [f"{layer_idx}.{w}" for w in self.self_attn._tied_weights_keys]

        if config.mlp_reduction:
            self._tied_weights_keys.append(f"{layer_idx}.mlp.up_proj.weight")

        bases = self.get_bases()
        if config.layer_reduction_type == "share" and layer_idx not in bases:
            self._tied_weights_keys += [
                f"{layer_idx}.mlp.gate_proj.weight",
                f"{layer_idx}.mlp.up_proj.weight",
                f"{layer_idx}.mlp.down_proj.weight",
            ]
        self._tied_weights_keys = list(set(self._tied_weights_keys))

    def get_bases(self):
        grouping = [
            int(i / self.config.num_hidden_layers * self.config.n_group)
            for i in range(self.config.num_hidden_layers)
        ]
        groups = defaultdict(list)
        for layer, group in enumerate(grouping):
            groups[group].append(layer)
        return [i[0] for _, i in groups.items()]


class SLLamaModel(LlamaModel,SLLamaPreTrainedModel):
    """Main SLLama transformer model."""

    _tied_weights_keys = []
    config_class = SLLamaConfig

    def __init__(self, config):
        super(SLLamaPreTrainedModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.local_hidden_size = config.hidden_size
        projector = NoneProjector

        if config.emb_reduction_type:
            self.local_hidden_size //= 4
            projector = PROJECTORS[config.emb_reduction_type]

        self.projector = projector(self.local_hidden_size, 4, config.hidden_act)
        self.embed_tokens = nn.Embedding(config.vocab_size, self.local_hidden_size, self.padding_idx)

        n_layer = config.num_hidden_layers
        if config.layer_reduction_type == "reuse":
            n_layer //= 2

        self.layers = nn.ModuleList([SLLamaDecoderLayer(config, i) for i in range(n_layer)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        if config.layer_reduction_type == "share":
            groups = self.group_layers()
            self.layer_share_weights(groups, part="all")
            self._tied_weights_keys += [f"layers.{w}" for w in self.layers[-1]._tied_weights_keys]

        if config.mlp_reduction:
            self.reduce_mlp_weight(self.layers)

        self._tied_weights_keys += list(
            set([f"layers.{w}" for layer in self.layers for w in layer._tied_weights_keys])
        )
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                cache_position=None, **flash_attn_kwargs):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # update use_cache as per reuse layer
        if self.config.layer_reduction_type == 'reuse':
            use_cache = False

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

            # handle potential new for projection
            inputs_embeds = self.projector(inputs_embeds)
            #print(inputs_embeds.shape)
        
        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False

        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)


        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        # cater for layer reuse
        num_reuse = 2 if self.config.layer_reduction_type == 'reuse' else 1
        for decoder_layer in self.layers:
            for _ in range(num_reuse):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                
                layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )
                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None

        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------


class SLLamaForCausalLM(SLLamaPreTrainedModel, LlamaForCausalLM):
    config_class = SLLamaConfig
    _tied_weights_keys = []

    def __init__(self, config):
        super(SLLamaPreTrainedModel, self).__init__(config)
        self.model = SLLamaModel(config)
        self._tied_weights_keys += [f"model.{w}" for w in self.model._tied_weights_keys]
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            self._tied_weights_keys.append("lm_head.weight")

        self.post_init()

class SLLamaForSequenceClassification(SLLamaPreTrainedModel, LlamaForSequenceClassification):
    config_class = SLLamaConfig

    def __init__(self, config):
        super(SLLamaPreTrainedModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.model = SLLamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()


# ---------------------------------------------------------------------------
# Auto-class registration
# ---------------------------------------------------------------------------

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

AutoConfig.register("sllama", SLLamaConfig)
AutoModel.register(SLLamaConfig, SLLamaModel)
AutoModelForCausalLM.register(SLLamaConfig, SLLamaForCausalLM)
AutoModelForSequenceClassification.register(SLLamaConfig, SLLamaForSequenceClassification)
