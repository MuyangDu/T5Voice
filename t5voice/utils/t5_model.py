# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# From: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

import copy
import math
from typing import Optional
from dataclasses import dataclass
import time
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.modeling_outputs import ModelOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5LayerNorm,
    T5DenseGatedActDense,
)
from .common_utils import (
    batched_top_k_filtering,
    batched_top_p_filtering
)
from .loss_utils import (
    AttentionCTCLoss
)
from tqdm import tqdm


@dataclass
class EncoderOutput(ModelOutput):
    hidden_states: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None
    self_attn_weights: torch.FloatTensor = None
    self_attn_scores: torch.FloatTensor = None
    cross_attn_weights: torch.FloatTensor = None
    cross_attn_scores: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None


@dataclass
class Seq2SeqLMOutput(ModelOutput):
    loss: torch.FloatTensor = None
    attn_ctc_loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    encoder_outputs: EncoderOutput = None
    decoder_outputs: EncoderOutput = None
    past_key_values: Optional[List[torch.FloatTensor]] = None


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        assert config.is_gated_act
        self.DenseReluDense = T5DenseGatedActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states).type_as(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values


    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        attention_prior=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        reset_past_key_value=False
    ):

        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        batch_size, seq_length = hidden_states.shape[:2]

        if past_key_value is not None:
            real_seq_length = seq_length + past_key_value[0].shape[2] if query_length is None else query_length
        else:
            real_seq_length = seq_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        query_states = self.q(hidden_states)
        if key_value_states is None:
            key_states, value_states = self.k(hidden_states), self.v(hidden_states)
            query_states, key_states, value_states = shape(query_states), shape(key_states), shape(value_states)
        elif past_key_value is None: # for decoder cross attn
            key_states, value_states = self.k(key_value_states), self.v(key_value_states)
            query_states, key_states, value_states = shape(query_states), shape(key_states), shape(value_states)
        else:
            query_states = shape(query_states)

        if past_key_value is not None:
            if key_value_states is None:
                past_key_states, past_value_states = past_key_value
                key_states = torch.cat([past_key_states, key_states], dim=2)
                value_states = torch.cat([past_value_states, value_states], dim=2)
            else:
                key_states, value_states = past_key_value

        if use_cache:
            present_key_value_state = (key_states, value_states)
        else:
            present_key_value_state = None

        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if attention_prior is not None:
            attention_prior = attention_prior.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            scores = torch.log_softmax(scores, dim=-1) + attention_prior

        attn_scores = scores.clone()

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            if past_key_value is not None or reset_past_key_value:
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                # Masking happens here, masked elements in the mask have the value of -inf
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
        
        position_bias_masked = position_bias

        scores += position_bias_masked

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights_dropped = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(torch.matmul(attn_weights_dropped, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        return (attn_output, present_key_value_state, position_bias, attn_weights, attn_scores)


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        past_key_value=None,
        use_cache=False
    ):
        normed_hidden_states = self.layer_norm(hidden_states).type_as(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs # hidden_states, present_key_value_state, position_bias, attn_weights, attn_scores


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        attention_prior=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        reset_past_key_value=False
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            attention_prior=attention_prior,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            reset_past_key_value=reset_past_key_value
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]
        return outputs # hidden_states, present_key_value_state, position_bias, attn_weights, attn_scores


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        cross_attention_prior=None,
        past_key_value=None,
        use_cache=False,
        reset_past_key_value=False
    ):

        if past_key_value is not None:
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
            if cross_attn_past_key_value[0] is None:
                cross_attn_past_key_value = None
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None
        
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache
        )
        hidden_states = self_attention_outputs[0]
        present_key_value_state = self_attention_outputs[1]
        attention_outputs = self_attention_outputs[2:]  # Relative position weights

        if self.is_decoder and encoder_hidden_states is not None:
            
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                attention_prior=cross_attention_prior,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                reset_past_key_value=reset_past_key_value
            )
            hidden_states = cross_attention_outputs[0]

            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        outputs = (hidden_states,)

        outputs = outputs + (present_key_value_state,) + attention_outputs

        return outputs  # hidden-states, present_key_value_state, (self-attention position bias), (self-attention attn weights), (self-attention attn scores), 
                        # (cross-attention position bias), (cross-attention attn weights), (cross-attention attn scores)


class T5Stack(nn.Module, ModuleUtilsMixin):
    def __init__(self, config, embed_tokens):
        super().__init__()
        assert embed_tokens is not None

        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )

        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    # @torch.compile(dynamic=True)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        cross_attention_prior=None,
        past_key_values=None,
        use_cache=False,
        reset_past_key_value=False
    ) -> EncoderOutput:
        input_shape = input_ids.size()
        
        if self.is_decoder:
            batch_size, seq_length, num_codebooks = input_shape
        else:
            batch_size, seq_length = input_shape
        
        if self.is_decoder:
            # If decoder, then input_ids has shape [batch_size, decoder_sequence_length, num_codebooks]
            inputs_embeds = [self.embed_tokens[index](input_ids[:, :, index]) for index in range(len(self.embed_tokens))]
            # [num_codebooks, batch_size, decoder_sequence_length, d_model]
            inputs_embeds = torch.stack(inputs_embeds)
            # [batch_size, decoder_sequence_length, num_codebooks, d_model]
            inputs_embeds = inputs_embeds.permute(1, 2, 0, 3)
            # [batch_size, decoder_sequence_length, d_model]
            inputs_embeds = inputs_embeds.sum(dim=2)
        else:
            # If encoder, then input_ids has shape [batch_size, encoder_sequence_length]
            inputs_embeds = self.embed_tokens(input_ids)

        if hasattr(self.config, 'is_bf16') and self.config.is_bf16:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)

        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        # Masking
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )
        
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape=(batch_size, seq_length))

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
            
        present_key_value_states = () if use_cache else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        
        layer_attn_weights = []
        layer_attn_scores = []
        layer_encoder_decoder_attn_weights = []
        layer_encoder_decoder_attn_scores = []

        for _, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                cross_attention_prior=cross_attention_prior,
                past_key_value=past_key_value,
                use_cache=use_cache,
                reset_past_key_value=reset_past_key_value
            )
            hidden_states = layer_outputs[0]
            present_key_value_state = layer_outputs[1]
            
            # We share the position biases between the layers - the first layer store them
            position_bias = layer_outputs[2]
            # attn_weights: batch_size x num_heads x sequence_length (encoder or decoder)
            attn_weights = layer_outputs[3]
            attn_scores = layer_outputs[4]
            layer_attn_weights.append(attn_weights)
            layer_attn_scores.append(attn_scores)

            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)
            
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[5]
                # encoder_decoder_attn_weights: batch_size x num_heads x decoder_sequence_length x encoder_sequence_length
                encoder_decoder_attn_weights = layer_outputs[6]
                encoder_decoder_attn_scores = layer_outputs[7]
                layer_encoder_decoder_attn_weights.append(encoder_decoder_attn_weights)
                layer_encoder_decoder_attn_scores.append(encoder_decoder_attn_scores)
        
        hidden_states = self.final_layer_norm(hidden_states).type_as(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # attn_weights: num_layers x batch_size x num_heads x sequence_length (encoder or decoder)
        attn_weights = torch.stack(layer_attn_weights)
        attn_scores = torch.stack(layer_attn_scores)
        # attn_weights: batch_size x num_layers x num_heads x sequence_length (encoder or decoder)
        attn_weights = attn_weights.permute(1, 0, 2, 3, 4)
        attn_scores = attn_scores.permute(1, 0, 2, 3, 4)
        
        if len(layer_encoder_decoder_attn_weights) > 0:
            # encoder_decoder_attn_weights: num_layers x batch_size x num_heads x decoder_sequence_length x encoder_sequence_length
            encoder_decoder_attn_weights = torch.stack(layer_encoder_decoder_attn_weights)
            encoder_decoder_attn_scores = torch.stack(layer_encoder_decoder_attn_scores)
            encoder_decoder_attn_weights = encoder_decoder_attn_weights.permute(1, 0, 2, 3, 4)
            encoder_decoder_attn_scores = encoder_decoder_attn_scores.permute(1, 0, 2, 3, 4)
        else:
            encoder_decoder_attn_weights = None
            encoder_decoder_attn_scores = None


        return EncoderOutput(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            self_attn_weights=attn_weights,
            self_attn_scores=attn_scores,
            cross_attn_weights=encoder_decoder_attn_weights,
            cross_attn_scores=encoder_decoder_attn_scores,
            past_key_values=present_key_value_states
        )


class T5Voice(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        
        config.is_encoder_decoder = False
        assert not config.tie_word_embeddings

        self.config = config
        self.model_dim = config.d_model
        self.encoder_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.num_decoder_special_tokens = 2 # decoder uses two special tokens: bos and eos
        self.decoder_vocab_size = self.num_decoder_special_tokens + config.codec_size
        self.decoder_emb = nn.ModuleList([
            nn.Embedding(self.decoder_vocab_size, config.d_model) for index in range(config.num_codebooks)
        ])
        self.decoder_bos_id = 0
        self.decoder_eos_id = 1

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = T5Stack(encoder_config, self.encoder_emb)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.decoder_emb)

        self.lm_heads = nn.ModuleList([
            nn.Linear(config.d_model, self.decoder_vocab_size, bias=False) for index in range(config.num_codebooks)
        ])

        self.generation_config = None

        self.apply(self._init_weights)
    
    # @torch.compile
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_prompt_input_ids: Optional[torch.LongTensor] = None,
        max_length = None,
        min_length = 10,
        temperature: Optional[torch.FloatTensor] = None,
        top_k: Optional[torch.LongTensor] = None,
        top_p: Optional[torch.FloatTensor] = None,
        use_cache = False,
        **kwargs,
    ) -> torch.LongTensor:
        
        B, _ = input_ids.size()
        
        if temperature is not None:
            temperature = temperature.reshape(B, 1, 1)
        if top_p is not None:
            top_p = top_p.reshape(B)
        if top_k is not None:
            top_k = top_k.reshape(B)
        
        if decoder_prompt_input_ids is None:
            decoder_input_ids = torch.zeros((B, 1, self.config.num_codebooks), dtype=torch.long, device=input_ids.device)
        else:
            decoder_input_ids = decoder_prompt_input_ids
            
        encoder_outputs = None
        
        complete = torch.zeros((B, 1), dtype=torch.long, device=input_ids.device)
        generated_valid_lengths = torch.zeros((B, 1), dtype=torch.long, device=input_ids.device)

        past_key_values = None
        decoder_output_ids = decoder_input_ids.clone()

        output_logits = None
        output_last_decoder_hiddens = None

        for _ in tqdm(range(max_length)):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                use_cache=use_cache,
                past_key_values=past_key_values
            )
            
            encoder_outputs = out.encoder_outputs
            decoder_outputs = out.decoder_outputs

            past_key_values = out.past_key_values

            # logits: [batch_size, num_codebooks, decoder_vocab_size]
            logits = out.logits[:, -1]

            if output_logits is None:
                output_logits = out.logits
            else:
                output_logits = torch.cat((output_logits,  out.logits[:, -1:]), dim=1)

            if output_last_decoder_hiddens is None:
                output_last_decoder_hiddens = decoder_outputs[0]
            else:
                output_last_decoder_hiddens = torch.cat((output_last_decoder_hiddens, decoder_outputs[0]), dim=1)
            
            if temperature is not None:
                logits = logits / temperature
            if top_p is not None:
                logits = batched_top_p_filtering(logits, top_p, min_tokens_to_keep=3)
            if top_k is not None:
                logits = batched_top_k_filtering(logits, top_k, min_tokens_to_keep=3)

            if generated_valid_lengths[0] < min_length:
                logits[:, :, 1] = float('-inf')
            
            scores = F.softmax(logits, dim=-1)
            # scores: [batch_size * num_codebooks, decoder_vocab_size]
            scores = scores.view(-1, self.decoder_vocab_size)
            # next_input_ids: [batch_size * num_codebooks, 1]
            next_input_ids = torch.multinomial(scores, num_samples=1)
            # next_input_ids: [batch_size, 1, num_codebooks]
            next_input_ids = next_input_ids.view(-1, 1, self.config.num_codebooks)
            
            decoder_output_ids = torch.cat([decoder_output_ids, next_input_ids], dim=1)

            if use_cache:
                decoder_input_ids = next_input_ids
            else:
                decoder_input_ids = torch.cat([decoder_input_ids, next_input_ids], dim=1)
            
            complete = complete | (next_input_ids == 1).any(dim=2)
            generated_valid_lengths += (1 - complete)
            
            if complete.all():
                break
        
        decoder_output_ids[:, -1, :] = 1
        # remove bos
        decoder_output_ids = decoder_output_ids[:, 1:, :]
        
        generated_valid_lengths -= 1

        return decoder_output_ids, generated_valid_lengths, encoder_outputs, decoder_outputs, output_logits, output_last_decoder_hiddens

    
    def generate_stream(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_prompt_input_ids: Optional[torch.LongTensor] = None,
        max_length = None,
        min_length = 10,
        temperature: Optional[torch.FloatTensor] = None,
        top_k: Optional[torch.LongTensor] = None,
        top_p: Optional[torch.FloatTensor] = None,
        use_cache = False,
        chunk_size = 50,
        **kwargs,
    ):
        B, _ = input_ids.size()

        assert B == 1
        
        if temperature is not None:
            temperature = temperature.reshape(B, 1, 1)
        if top_p is not None:
            top_p = top_p.reshape(B)
        if top_k is not None:
            top_k = top_k.reshape(B)
        
        if decoder_prompt_input_ids is None:
            decoder_input_ids = torch.zeros((B, 1, self.config.num_codebooks), dtype=torch.long, device=input_ids.device)
        else:
            decoder_input_ids = decoder_prompt_input_ids
            
        encoder_outputs = None
        
        complete = torch.zeros((B, 1), dtype=torch.long, device=input_ids.device)
        generated_valid_lengths = torch.zeros((B, 1), dtype=torch.long, device=input_ids.device)

        past_key_values = None
        decoder_output_ids = decoder_input_ids.clone()

        output_logits = None
        output_last_decoder_hiddens = None

        chunk_codec = []

        for _ in tqdm(range(max_length)):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                use_cache=use_cache,
                past_key_values=past_key_values
            )
            
            encoder_outputs = out.encoder_outputs
            decoder_outputs = out.decoder_outputs

            past_key_values = out.past_key_values

            # logits: [batch_size, num_codebooks, decoder_vocab_size]
            logits = out.logits[:, -1]

            if output_logits is None:
                output_logits = out.logits
            else:
                output_logits = torch.cat((output_logits,  out.logits[:, -1:]), dim=1)

            if output_last_decoder_hiddens is None:
                output_last_decoder_hiddens = decoder_outputs[0]
            else:
                output_last_decoder_hiddens = torch.cat((output_last_decoder_hiddens, decoder_outputs[0]), dim=1)
            
            if temperature is not None:
                logits = logits / temperature
            if top_p is not None:
                logits = batched_top_p_filtering(logits, top_p, min_tokens_to_keep=3)
            if top_k is not None:
                logits = batched_top_k_filtering(logits, top_k, min_tokens_to_keep=3)

            if generated_valid_lengths[0] < min_length:
                logits[:, :, 1] = float('-inf')
            
            scores = F.softmax(logits, dim=-1)
            # scores: [batch_size * num_codebooks, decoder_vocab_size]
            scores = scores.view(-1, self.decoder_vocab_size)
            # next_input_ids: [batch_size * num_codebooks, 1]
            next_input_ids = torch.multinomial(scores, num_samples=1)
            # next_input_ids: [batch_size, 1, num_codebooks]
            next_input_ids = next_input_ids.view(-1, 1, self.config.num_codebooks)
            
            decoder_output_ids = torch.cat([decoder_output_ids, next_input_ids], dim=1)

            if use_cache:
                decoder_input_ids = next_input_ids
            else:
                decoder_input_ids = torch.cat([decoder_input_ids, next_input_ids], dim=1)
            
            complete = complete | (next_input_ids == 1).any(dim=2)
            generated_valid_lengths += (1 - complete)

            chunk_codec.append(next_input_ids.clone())
            if len(chunk_codec) == chunk_size:
                if complete.all():
                    chunk_codec = chunk_codec[:-1]
                    valid_length = chunk_size - 1
                    end = True
                else:
                    valid_length = chunk_size
                    end = False
                if valid_length > 0:
                    chunk_to_yield = torch.concat(chunk_codec, dim=1)
                    yield chunk_to_yield, valid_length, end
                chunk_codec.clear()
            
            if complete.all():
                break
            
        if len(chunk_codec) > 0:
            chunk_codec = chunk_codec[:-1]
            valid_length = len(chunk_codec)
            end = True
            if valid_length > 0:
                chunk_to_yield = torch.concat(chunk_codec, dim=1)
                yield chunk_to_yield, valid_length, end
    

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_outputs = None,
        cross_attention_prior=None,
        input_lengths = None,
        decoder_input_lengths = None,
        past_key_values = None,
        use_cache = False
    ) -> Seq2SeqLMOutput:
        
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden_states = encoder_outputs.hidden_states
        encoder_self_attn_weights = encoder_outputs.self_attn_weights
        encoder_self_attn_scores = encoder_outputs.self_attn_scores

        if cross_attention_prior is not None:
            cross_attention_prior = torch.log(cross_attention_prior + 1e-8)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            cross_attention_prior=cross_attention_prior,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        sequence_output = decoder_outputs[0]
        decoder_self_attn_weights = decoder_outputs.self_attn_weights
        decoder_cross_attn_weights = decoder_outputs.cross_attn_weights
        decoder_self_attn_scores = decoder_outputs.self_attn_scores
        decoder_cross_attn_scores = decoder_outputs.cross_attn_scores
        
        lm_logits = [lm_head(sequence_output) for lm_head in self.lm_heads]
        # lm_logits: [num_codebooks, batch_size, decoder_seq_length, codec_size]
        lm_logits = torch.stack(lm_logits)
        # lm_logits: [batch_size, decoder_seq_length, num_codebooks, codec_size]
        lm_logits = lm_logits.permute(1, 2, 0, 3)

        loss = None
        attn_ctc_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.reshape(-1, lm_logits.size(-1)), labels.reshape(-1))
            averaged_decoder_cross_attn_scores = torch.mean(decoder_cross_attn_scores, dim=[1, 2])
            attn_ctc_loss_fct = AttentionCTCLoss()
            attn_ctc_loss = attn_ctc_loss_fct(
                attn_scores=averaged_decoder_cross_attn_scores, 
                in_lens=input_lengths.squeeze(), 
                out_lens=decoder_input_lengths.squeeze()
            )

        return Seq2SeqLMOutput(
            loss=loss,
            attn_ctc_loss=attn_ctc_loss,
            logits=lm_logits,
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
            past_key_values=decoder_outputs.past_key_values
        )

    def _init_weights(self, module):
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Voice)):
            module.encoder_emb.weight.data.normal_(mean=0.0, std=factor * 1.0)
            for index in range(len(module.decoder_emb)):
                module.decoder_emb[index].weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_heads"):
                for index in range(len(module.lm_heads)):
                    module.lm_heads[index].weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseGatedActDense):
            d_ff, d_model = module.wi_0.weight.data.size()
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((d_ff) ** -0.5))
        elif isinstance(module, T5Attention):
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if hasattr(module, "relative_attention_bias"):
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))