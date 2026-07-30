# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by the MoRE authors: added learnable per-layer depth embeddings
# applied before routing (config: use_depth_embedding) and cross-layer expert
# parameter tying (config: grouping_factor, tying_strategy).

from typing import Callable, Optional, Union, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers.utils.generic import check_model_inputs

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import OutputRecorder
from .configuration_mixtral import MixtralConfig

# --- Sonic MoE Imports ---
try:
    from sonicmoe import MoE, KernelBackendMoE
    from sonicmoe.enums import ActivationType
    _SONIC_AVAILABLE = True
except ImportError:
    _SONIC_AVAILABLE = False

class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

class MixtralSparseMoeBlock(nn.Module):
    """Manual Iterative Implementation (Slow, Reference)"""
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
            
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Safe Norm
        routing_weights /= (routing_weights.sum(dim=-1, keepdim=True) + 1e-9)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states_flat)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0: continue

            current_state = hidden_states_flat[top_x]
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            
        return final_hidden_states.view(batch_size, sequence_length, hidden_dim), router_logits



class MissingSonicMoE(ImportError):
    """Raised when a config asks for the SonicMoE kernel but it is not installed.

    Design decision: the message names the workaround, not just the problem.
    SonicMoE requires a Hopper GPU (H100/H200) and CUDA 12.9+, so most readers
    will not have it — and most readers only want to inspect the architecture,
    which the reference MixtralSparseMoeBlock supports on CPU. Subclasses
    ImportError so existing `except ImportError` handlers still catch it.
    """


class MixtralSonicMoeBlock(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        if not _SONIC_AVAILABLE:
            raise MissingSonicMoE(
                "This config sets use_scattermoe=True, which needs the SonicMoE "
                "kernel (Hopper GPU, CUDA 12.9+). It is not installed.\n"
                "  To inspect the model on CPU: set use_scattermoe=False to use "
                "the reference MixtralSparseMoeBlock.\n"
                "  To train: run setup.sh on an H100/H200 machine."
            )

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        
        self.moe = MoE(
            num_experts=self.num_experts,
            num_experts_per_tok=self.top_k,
            hidden_size=self.hidden_dim,
            intermediate_size=self.ffn_dim,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=config.initializer_range,
        )

    #no compile
    @torch.compiler.disable
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [Batch, Seq, Hidden]
        original_dtype = hidden_states.dtype

        # Sonic kernels require BF16 for both activations AND weights
        hidden_states = hidden_states.to(torch.bfloat16)

        # Lazy one-time conversion of MoE weights to BF16
        if self.moe.c_fc.weight.dtype != torch.bfloat16:
            self.moe = self.moe.to(torch.bfloat16)

        # Forward pass
        # Note: Sonic returns (output, aux_loss), NOT (output, logits)
        output, aux_loss = self.moe(
            hidden_states,
            kernel_backend_moe=KernelBackendMoE.sonicmoe
        )

        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        return output, aux_loss

@use_kernel_forward_from_hub("RMSNorm")
class MixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MixtralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class MixtralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MixtralRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: MixtralConfig, device=None):
        super().__init__()
        # Resolve rope_type from rope_parameters dict or legacy rope_scaling
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            self.rope_type = config.rope_parameters.get("rope_type", "default")
        elif hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        if self.rope_type == "default":
            inv_freq, self.attention_scaling = self.compute_default_rope_parameters(config, device)
        else:
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        base = config.rope_theta
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)



class MixtralDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MixtralAttention(config, layer_idx)

        # Flag now controls Sonic MoE
        self.use_scattermoe = config.use_scattermoe
        if config.use_scattermoe:
            self.mlp = MixtralSonicMoeBlock(config)
        else:
            self.mlp = MixtralSparseMoeBlock(config)
            
        self.input_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        depth_emb: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply Depth Embedding if present
        if depth_emb is not None:
            hidden_states = hidden_states + depth_emb
            
        # Run MoE Block
        # If Sonic: router_logits is actually the scalar aux_loss
        # If Manual: router_logits is the actual tensor logits
        #print(f"DEBUG Decoder Dtypes: hidden_states: {hidden_states.dtype}")
        hidden_states, router_logits = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states

        return hidden_states, router_logits


@auto_docstring
class MixtralPreTrainedModel(PreTrainedModel):
    config: MixtralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MixtralDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(MixtralSparseMoeBlock, index=1),
        "hidden_states": MixtralDecoderLayer,
        "attentions": MixtralAttention,
    }

@auto_docstring
class MixtralModel(MixtralPreTrainedModel):
    def __init__(self, config: MixtralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Initialize Depth Embeddings
        self.use_depth_embedding = config.use_depth_embedding
        if self.use_depth_embedding:
            self.all_depth_embeddings = nn.Parameter(
                torch.empty(config.num_hidden_layers, config.hidden_size)
            )
            nn.init.xavier_uniform_(self.all_depth_embeddings)
        else:
            self.all_depth_embeddings = None

        self.layers = nn.ModuleList(
            [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MixtralRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.use_scattermoe = config.use_scattermoe
        self.group_size = config.grouping_factor
        self.tying_strategy = config.tying_strategy
        self.post_init()

    def tie_weights(self, *args, **kwargs):
        super().tie_weights(*args, **kwargs)
        if self.group_size <= 1:
            return
        num_layers = self.config.num_hidden_layers
        num_unique_sets = num_layers // self.group_size
        tied_map = {}

        for i in range(num_layers):
            if self.tying_strategy == "block":
                leader_idx = (i // self.group_size) * self.group_size
            elif self.tying_strategy == "strided":
                leader_idx = i % num_unique_sets
            else:
                raise ValueError(f"Unknown tying_strategy: {self.tying_strategy}")

            if i != leader_idx:
                leader_moe = self.layers[leader_idx].mlp
                curr_moe = self.layers[i].mlp

                if self.use_scattermoe:
                    curr_moe.moe.c_fc.weight = leader_moe.moe.c_fc.weight
                    curr_moe.moe.c_proj.weight = leader_moe.moe.c_proj.weight
                    for name in ("moe.c_fc.weight", "moe.c_proj.weight"):
                        tied_map[f"layers.{i}.mlp.{name}"] = \
                            f"layers.{leader_idx}.mlp.{name}"
                else:
                    for j in range(len(curr_moe.experts)):
                        curr_moe.experts[j].w1.weight = leader_moe.experts[j].w1.weight
                        curr_moe.experts[j].w2.weight = leader_moe.experts[j].w2.weight
                        curr_moe.experts[j].w3.weight = leader_moe.experts[j].w3.weight
                        for wn in ("w1.weight", "w2.weight", "w3.weight"):
                            tied_map[f"layers.{i}.mlp.experts.{j}.{wn}"] = \
                                f"layers.{leader_idx}.mlp.experts.{j}.{wn}"

        # Register tied keys so HF save/load handles them correctly
        # _tied_weights_keys is read by _get_tied_weight_keys() during save_pretrained
        self._tied_weights_keys = tied_map

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_router_logits = []
        
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            
            depth_emb = None
            if self.use_depth_embedding and self.all_depth_embeddings is not None:
                depth_emb = self.all_depth_embeddings[layer_idx]

            hidden_states, router_logits = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                depth_emb=depth_emb,
                **kwargs,
            )

            # Store router logits (or scalar loss if Sonic)
            all_router_logits.append(router_logits)
            
        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            router_logits=tuple(all_router_logits),
        )

def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Standard load balancing loss for Manual implementation.
    Not used when Sonic MoE is active (as Sonic returns pre-computed loss).
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )
    
    overall_loss = torch.sum(router_prob_per_expert * tokens_per_expert)
    return overall_loss * num_experts


def compute_grouped_switch_loss(
    layers: nn.ModuleList,
    num_experts: int,
    group_size: int,
    tying_strategy: str,
    num_layers: int,
) -> torch.Tensor:
    """Switch loss pooled across layers that share expert weights.

    Instead of penalizing per-layer imbalance independently, this pools
    routing statistics (acc_probs, expert_frequency) across layers in the
    same tying group before computing the loss.  This allows routers to
    depth-partition experts without penalty, as long as the group-level
    utilization is balanced.
    """
    if tying_strategy == "block":
        groups = [range(g, min(g + group_size, num_layers))
                  for g in range(0, num_layers, group_size)]
    elif tying_strategy == "strided":
        num_unique = num_layers // group_size
        groups = [range(i, num_layers, num_unique) for i in range(num_unique)]
    else:
        raise ValueError(f"Unknown tying_strategy: {tying_strategy}")

    total_loss = None
    for group in groups:
        pooled_probs = sum(layers[l].mlp.moe._routing_acc_probs for l in group)
        pooled_freq = sum(layers[l].mlp.moe._routing_expert_freq for l in group)

        group_loss = num_experts * (
            F.normalize(pooled_probs, p=1, dim=0) *
            F.normalize(pooled_freq, p=1, dim=0)
        ).sum()

        total_loss = group_loss if total_loss is None else total_loss + group_loss

    return total_loss


def compute_all_grouped_switch_loss(
    layers: nn.ModuleList,
    num_experts: int,
    num_layers: int,
) -> torch.Tensor:
    """Switch loss pooled across all layers by expert index.

    Pools routing statistics (acc_probs, expert_frequency) over all layers
    so that each expert index is one group. Encourages each expert index
    to be used equally in aggregate across depth; allows per-layer
    specialization without penalty.
    *Note: This is actually the official hugginface implementation of the 
    load balancing loss, but it is not correct for switch loss (per layer).
    """
    pooled_probs = sum(layers[l].mlp.moe._routing_acc_probs for l in range(num_layers))
    pooled_freq = sum(layers[l].mlp.moe._routing_expert_freq for l in range(num_layers))
    return num_experts * (
        F.normalize(pooled_probs, p=1, dim=0) * F.normalize(pooled_freq, p=1, dim=0)
    ).sum()


@auto_docstring
class MixtralForCausalLM(MixtralPreTrainedModel, GenerationMixin):
    _tied_weights_keys = []
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = MixtralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.use_grouped_aux_loss = getattr(config, "use_grouped_aux_loss", False)
        self.all_grouped_aux_loss = getattr(config, "all_grouped_aux_loss", False)
        self.group_size = config.grouping_factor
        self.tying_strategy = config.tying_strategy

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self, *args, **kwargs):
        super().tie_weights(*args, **kwargs)
        self.model.tie_weights(*args, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            # Check if we are using Sonic MoE (scalar losses) or Manual (tensor logits)
            # Sonic returns a tuple of scalar tensors representing loss per layer
            first_layer_out = outputs.router_logits[0]
            
            if first_layer_out.dim() == 0 or (first_layer_out.dim() == 1 and first_layer_out.numel() == 1):
                # Sonic Path
                if self.all_grouped_aux_loss:
                    # All-grouped loss: pool routing stats across all layers by expert index
                    aux_loss = compute_all_grouped_switch_loss(
                        self.model.layers,
                        self.num_experts,
                        self.config.num_hidden_layers,
                    )
                elif self.use_grouped_aux_loss and self.group_size > 1:
                    # Grouped loss: pool routing stats across tied layers
                    aux_loss = compute_grouped_switch_loss(
                        self.model.layers,
                        self.num_experts,
                        self.group_size,
                        self.tying_strategy,
                        self.config.num_hidden_layers,
                    )
                else:
                    # Per-layer loss: sum pre-calculated scalar losses
                    aux_loss = sum(outputs.router_logits)
            else:
                # Manual Path: Calculate loss from logits
                aux_loss = load_balancing_loss_func(
                    outputs.router_logits,
                    self.num_experts,
                    self.num_experts_per_tok,
                    attention_mask,
                )
            
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class MixtralForSequenceClassification(GenericForSequenceClassification, MixtralPreTrainedModel):
    pass


class MixtralForTokenClassification(GenericForTokenClassification, MixtralPreTrainedModel):
    pass


class MixtralForQuestionAnswering(GenericForQuestionAnswering, MixtralPreTrainedModel):
    pass


__all__ = [
    "MixtralForCausalLM",
    "MixtralForQuestionAnswering",
    "MixtralModel",
    "MixtralPreTrainedModel",
    "MixtralForSequenceClassification",
    "MixtralForTokenClassification",
]