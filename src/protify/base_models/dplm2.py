"""
Barebones DPLM2 integration for Protify.

This implementation focuses on sequence embedding and masked-LM scoring paths.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import EsmTokenizer
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    ModelOutput,
)
from transformers.models.esm.modeling_esm import (
    EsmAttention,
    EsmEmbeddings,
    EsmEncoder,
    EsmForMaskedLM,
    EsmForSequenceClassification,
    EsmForTokenClassification,
    EsmIntermediate,
    EsmLayer,
    EsmLMHead,
    EsmModel,
    EsmOutput,
    EsmPooler,
    EsmPreTrainedModel,
    EsmSelfAttention,
    EsmSelfOutput,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    logger,
)

from .base_tokenizer import BaseSequenceTokenizer


MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


@dataclass
class DPLM2MaskedLMOutput(ModelOutput):
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class ModifiedRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.aa_type = 1
        self.struct_type = 0

    def _has_multimodal_tokens(self, type_ids: Optional[torch.Tensor]) -> bool:
        if type_ids is None:
            return False
        aa_present = (type_ids == self.aa_type).any()
        struct_present = (type_ids == self.struct_type).any()
        return bool(aa_present and struct_present)

    def _update_cos_sin_tables(
        self,
        x: torch.Tensor,
        type_ids: Optional[torch.Tensor],
        seq_dimension: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[seq_dimension]
        if self._has_multimodal_tokens(type_ids):
            seq_len = seq_len // 2

        cache_is_stale = (
            self._cos_cached is None
            or self._sin_cached is None
            or seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
        )
        if cache_is_stale:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        type_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, type_ids=type_ids, seq_dimension=-2
        )

        if self._has_multimodal_tokens(type_ids):
            q_1, q_2 = q.chunk(2, dim=-2)
            k_1, k_2 = k.chunk(2, dim=-2)
            q_1 = apply_rotary_pos_emb(q_1, self._cos_cached, self._sin_cached)
            q_2 = apply_rotary_pos_emb(q_2, self._cos_cached, self._sin_cached)
            k_1 = apply_rotary_pos_emb(k_1, self._cos_cached, self._sin_cached)
            k_2 = apply_rotary_pos_emb(k_2, self._cos_cached, self._sin_cached)
            return torch.cat((q_1, q_2), dim=-2), torch.cat((k_1, k_2), dim=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class ModifiedEsmSelfAttention(EsmSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.rotary_embeddings = ModifiedRotaryEmbedding(
            dim=self.attention_head_size
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        type_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        if past_key_values is not None:
            past_key_value = past_key_values

        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(encoder_hidden_states)
            )
            value_layer = self.transpose_for_scores(
                self.value(encoder_hidden_states)
            )
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer = query_layer * self.attention_head_size**-0.5

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(
                query_layer, key_layer, type_ids
            )

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            raise NotImplementedError

        if head_mask is not None and torch.is_tensor(head_mask):
            query_layer = query_layer * head_mask

        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()

        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            scale=1.0,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,
        )
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, None)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class ModifiedEsmAttention(EsmAttention):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.self = ModifiedEsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        type_ids=None,
    ):
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            type_ids,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ModifiedEsmLayer(EsmLayer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ModifiedEsmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = ModifiedEsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        type_ids=None,
    ):
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            type_ids=type_ids,
        )
        attention_output = self_attention_outputs[0]

        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not self.add_cross_attention:
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]

            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = (
                present_key_value + cross_attn_present_key_value
            )

        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs

        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs


class ModifiedEsmEncoder(EsmEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.layer = nn.ModuleList(
            [ModifiedEsmLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.emb_layer_norm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        type_ids=None,
    ):
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            ()
            if output_attentions and self.config.add_cross_attention
            else None
        )

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = (
                past_key_values[i] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    type_ids,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    type_ids,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        layer_outputs[2],
                    )

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class ModifiedEsmModel(EsmModel):
    config_class = EsmModel.config_class
    all_tied_weights_keys = {}

    def __init__(self, config, add_pooling_layer=True):
        EsmPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = EsmEmbeddings(config)
        self.encoder = ModifiedEsmEncoder(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.post_init()

    def _convert_head_mask_to_5d(
        self, head_mask: torch.Tensor, num_hidden_layers: int
    ) -> torch.Tensor:
        if head_mask.dim() == 1:
            head_mask = (
                head_mask.unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert (
            head_mask.dim() == 5
        ), f"head_mask.dim != 5, got {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    def get_head_mask(
        self,
        head_mask: Optional[torch.Tensor],
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> Union[torch.Tensor, List[None]]:
        if head_mask is None:
            return [None] * num_hidden_layers
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions
    ]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = (
                use_cache if use_cache is not None else self.config.use_cache
            )
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        batch_size, seq_length = input_shape
        device = (
            input_ids.device if input_ids is not None else inputs_embeds.device
        )

        past_key_values_length = (
            past_key_values[0][0].shape[2]
            if past_key_values is not None
            else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)),
                device=device,
            )

        if attention_mask.dim() == 4:
            extended_attention_mask = attention_mask
        else:
            extended_attention_mask: torch.Tensor = (
                self.get_extended_attention_mask(attention_mask, input_shape)
            )

        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size,
                encoder_sequence_length,
            )
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = encoder_attention_mask

        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers
        )

        if input_ids is not None:
            embedding_attention_mask = input_ids.ne(self.config.pad_token_id)
        else:
            embedding_attention_mask = attention_mask

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=embedding_attention_mask,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            type_ids=type_ids,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@register_model("dplm2_esm")
class EsmForDPLM2(EsmForMaskedLM):
    config_class = EsmForMaskedLM.config_class
    all_tied_weights_keys = {}

    def __init__(self, config, dropout=0.1, vocab_size=None):
        config.hidden_dropout_prob = dropout
        config.tie_word_embeddings = False
        if vocab_size is not None:
            config.vocab_size = vocab_size

        EsmPreTrainedModel.__init__(self, config)
        self.esm = ModifiedEsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        self.init_weights()
        self.pad_id = config.pad_token_id
        self.contact_head = None

    def _get_modality_type(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_mask = attention_mask.bool()
        modality_type = ((input_ids < 33) & input_mask).int()
        pad_type = 2
        modality_type[~input_mask] = pad_type
        return modality_type

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        type_ids=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if attention_mask is None:
            assert input_ids is not None
            attention_mask = input_ids.ne(self.pad_id)

        if type_ids is None:
            assert input_ids is not None
            type_ids = self._get_modality_type(input_ids, attention_mask)

        outputs = self.esm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            type_ids=type_ids,
        )

        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        return DPLM2MaskedLMOutput(
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


presets = {
    "DPLM2-150": "airkingbd/dplm2_150m",
    "DPLM2-650": "airkingbd/dplm2_650m",
    "DPLM2-3B": "airkingbd/dplm2_3b",
}


class DPLM2TokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: EsmTokenizer):
        super().__init__(tokenizer)

    def __call__(
        self, sequences: Union[str, List[str]], **kwargs
    ) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", "longest")
        kwargs.setdefault("add_special_tokens", True)
        tokenized = self.tokenizer(sequences, **kwargs)
        return tokenized


class DPLM2ForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.dplm2 = EsmForDPLM2.from_pretrained(model_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        out = self.dplm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if output_attentions:
            return out.last_hidden_state, out.attentions
        return out.last_hidden_state


def get_dplm2_tokenizer(preset: str):
    return DPLM2TokenizerWrapper(
        EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    )


def build_dplm2_model(preset: str, masked_lm: bool = False, **kwargs):
    model_path = presets[preset]
    if masked_lm:
        model = EsmForDPLM2.from_pretrained(model_path).eval()
    else:
        model = DPLM2ForEmbedding(model_path).eval()
    tokenizer = get_dplm2_tokenizer(preset)
    return model, tokenizer


def get_dplm2_for_training(
    preset: str,
    tokenwise: bool = False,
    num_labels: int = None,
    hybrid: bool = False,
):
    model_path = presets[preset]
    if hybrid:
        model = EsmForDPLM2.from_pretrained(model_path).eval()
    else:
        if tokenwise:
            model = EsmForTokenClassification.from_pretrained(
                model_path, num_labels=num_labels
            ).eval()
        else:
            model = EsmForSequenceClassification.from_pretrained(
                model_path, num_labels=num_labels
            ).eval()
    tokenizer = get_dplm2_tokenizer(preset)
    return model, tokenizer


if __name__ == "__main__":
    # py -m src.protify.base_models.dplm2
    model, tokenizer = build_dplm2_model("DPLM2-150")
    print(model)
    print(tokenizer)
    print(
        tokenizer(
            "MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL"
        )
    )
