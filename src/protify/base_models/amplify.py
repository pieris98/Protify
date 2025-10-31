import torch
import torch.nn as nn
import safetensors
from typing import Optional, Tuple, Union, List, Dict
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    AutoModel,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM
)
from transformers.modeling_outputs import MaskedLMOutput
from .base_tokenizer import BaseSequenceTokenizer
from .amplify_utils import (
    SwiGLU,
    RMSNorm,
    apply_rotary_emb,
    memory_efficient_attention,
    precompute_freqs_cis,
)
from huggingface_hub import hf_hub_download
import json
    
presets = {
    'AMPLIFY-120': 'GleghornLab/AMPLIFY_120M',
    'AMPLIFY-350': 'GleghornLab/AMPLIFY_350M',
}


def _attention_mask_to_additive(
    attention_mask: Optional[torch.Tensor],
    target_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Convert a standard attention mask (1 for tokens, 0 for padding) into an additive mask.

    The additive mask uses 0 for valid tokens and a large negative value for padding positions,
    as expected by AMPLIFY. If the incoming mask already contains negative values, it is cast to the 
    requested dtype and returned as is.
    """

    if attention_mask is None:
        return None

    # If the mask already contains negative values, assume it is already additive
    if attention_mask.dtype.is_floating_point and attention_mask.min().item() < 0:
        return attention_mask.to(dtype=target_dtype)

    bool_mask = attention_mask.to(torch.bool)
    additive_mask = torch.zeros_like(attention_mask, dtype=target_dtype)

    if not torch.all(bool_mask):
        additive_mask = additive_mask.masked_fill(~bool_mask, torch.finfo(additive_mask.dtype).min)

    return additive_mask

class AMPLIFYConfig(PretrainedConfig):
    model_type = "AMPLIFY"
    # All config parameters must have a default value
    def __init__(
        self,
        hidden_size: int = 960,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 15,
        intermediate_size: int = 3840,
        dropout_prob: float = 0,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        rms_norm: bool = True,
        norm_eps: float = 1e-05,
        hidden_act: str = "SwiGLU",
        layer_norm_after_embedding: bool = False,
        layer_norm_before_last_layer: bool = True,
        vocab_size: int = 27,
        ffn_bias: bool = False,
        att_bias: bool = False,
        pad_token_id: int = 0,
        max_length: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_prob = dropout_prob
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.rms_norm = rms_norm
        self.norm_eps = norm_eps
        self.hidden_act = hidden_act
        self.layer_norm_after_embedding = layer_norm_after_embedding
        self.layer_norm_before_last_layer = layer_norm_before_last_layer
        self.vocab_size = vocab_size
        self.ffn_bias = ffn_bias
        self.att_bias = att_bias
        self.pad_token_id = pad_token_id
        self.max_length = max_length

class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: AMPLIFYConfig):
        """Initialize a EncoderBlock.

        Args:
            hidden_size (int): _description_
            num_attention_heads (int): _description_
            intermediate_size (int, optional): _description_. Defaults to 2048.
            dropout_prob (float, optional): _description_. Defaults to 0.1.
            activation (str, optional): _description_. Defaults to "relu".
            rms_norm (bool, optional): _description_. Defaults to True.
            norm_eps (float, optional): _description_. Defaults to 1e-5.
            pad_token_id (int, optional): _description_. Defaults to 0.
            max_length (int, optional): _description_. Defaults to 2048.
            ffn_bias (bool, optional): _description_. Defaults to False.
            att_bias (bool, optional): _description_. Defaults to False.
        """
        super().__init__()

        self.config = config
        self.d_head = config.hidden_size // config.num_attention_heads

        # Attention
        self.q = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=config.att_bias)
        self.k = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=config.att_bias)
        self.v = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=config.att_bias)
        self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=config.att_bias)
        self.resid_dropout = nn.Dropout(config.dropout_prob)

        # Feedforward network
        act = config.hidden_act.lower()
        if act == "swiglu":
            # To keep the number of parameters and the amount of computation constant, we reduce the number of
            # hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
            # avoid RuntimeError due to misaligned operand
            multiple_of = 8
            intermediate_size = int(2 * config.intermediate_size / 3)
            intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
            self.ffn = SwiGLU(
                config.hidden_size,
                intermediate_size,
                config.hidden_size,
                bias=config.ffn_bias
            )
        elif act == "relu":
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size, bias=config.ffn_bias),
                nn.ReLU(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=config.ffn_bias),
            )
        elif act == "gelu":
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size, bias=config.ffn_bias),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=config.ffn_bias),
            )
        else:
            raise ValueError(f"Unsupported hidden_act: {config.hidden_act}")

        self.attention_norm = RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)

        self.ffn_dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor, output_attentions: bool):
        attn, contact = self._att_block(self.attention_norm(x), pad_mask, freqs_cis, output_attentions)
        x = x + attn
        x = x + self._ff_block(self.ffn_norm(x))
        return x, contact

    def _att_block(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor, output_attentions: bool):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)

        # Reshape for rotary embeddings
        xq = xq.view(batch_size, seq_len, self.config.num_attention_heads, self.d_head)
        xk = xk.view(batch_size, seq_len, self.config.num_attention_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.config.num_attention_heads, self.d_head)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        attn = memory_efficient_attention(
            query=xq,
            key=xk,
            value=xv,
            attn_bias=pad_mask,
            p=self.config.dropout_prob if self.training else 0,
        )

        _attn = None
        if output_attentions:
            _attn = xq.permute(0, 2, 1, 3) @ xk.permute(0, 2, 3, 1) / (xq.size(-1) ** 0.5)
            if pad_mask is not None:
                _attn = _attn + pad_mask
            _attn = _attn.softmax(-1)
        return self.resid_dropout(self.wo(attn.view(batch_size, seq_len, self.config.num_attention_heads * self.d_head))), _attn

    def _ff_block(self, x: torch.Tensor):
        return self.ffn_dropout(self.ffn(x))


class AMPLIFYPreTrainedModel(PreTrainedModel):
    config_class = AMPLIFYConfig

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-self.config.embedding_init_range, self.config.embedding_init_range)


class AMPLIFY(AMPLIFYPreTrainedModel):
    """The main model class.

       Args:
          config (amplify.model.amplify.AMPLIFYConfig): model configuration, usually defined from the Hydra configuration.
    """
    def __init__(self, config: AMPLIFYConfig, **kwargs):
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if config.layer_norm_after_embedding:
            self.layer_norm_1 = RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        if config.layer_norm_before_last_layer:
            self.layer_norm_2 = RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, src, pad_mask=None, output_hidden_states=False, output_attentions=False):
        # Initialize
        hidden_states, attentions = [], []

        # Expand and repeat: (Batch, Length) -> (Batch, Heads, Length, Length)
        if pad_mask is not None and not torch.all(pad_mask == 0):
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.config.num_attention_heads, pad_mask.size(-1), 1)
        else:
            pad_mask = None

        # RoPE
        self.freqs_cis = self.freqs_cis.to(src.device, non_blocking=True)
        freqs_cis = self.freqs_cis[: src.shape[1]]

        # Embedding
        x = self.encoder(src)
        if self.config.layer_norm_after_embedding:
            x = self.layer_norm_1(x)

        # Transformer encoder
        for layer in self.transformer_encoder:
            x, attn = layer(x, pad_mask, freqs_cis, output_attentions)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(attn)

        # Classification head with layer norm
        logits = self.decoder(self.layer_norm_2(x) if self.config.layer_norm_before_last_layer else x)

        # Return logits or the output of the last hidden layer
        return MaskedLMOutput(logits=logits, hidden_states=hidden_states, attentions=attentions)


class AmplifyTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'longest')
        kwargs.setdefault('add_special_tokens', True)
        tokenized = self.tokenizer(sequences, **kwargs)
        return tokenized


class AmplifyForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        # Load config from HuggingFace
        config_file = hf_hub_download(repo_id=model_path, filename="config.json")
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        config = AMPLIFYConfig(**config_dict)
        self.plm = AMPLIFY(config)

        weight_file = hf_hub_download(repo_id=model_path, filename="model.safetensors")
        state_dict = safetensors.torch.load_file(weight_file)
        self.plm.load_state_dict(state_dict)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
    ) -> torch.Tensor:
        if attention_mask is not None:
            pad_mask = _attention_mask_to_additive(attention_mask, self.plm.encoder.weight.dtype)
        else:
            pad_mask = None

        out = self.plm(
            src=input_ids,
            pad_mask=pad_mask,
            output_attentions=output_attentions if output_attentions is not None else False,
            output_hidden_states=output_hidden_states,
        )
        residue_embeddings = out.hidden_states[-1]

        if attention_mask is not None:
            mask = attention_mask.to(torch.bool).unsqueeze(-1)
            residue_embeddings = residue_embeddings.masked_fill(~mask, 0.0)

        if output_attentions:
            return residue_embeddings, out.attentions
        else:
            return residue_embeddings


class AmplifyForMaskedLM(nn.Module):
    """Wrapper for AMPLIFY model to use for Masked Language Modeling tasks."""
    def __init__(self, model_path: str):
        super().__init__()
        # Load config from HuggingFace
        config_file = hf_hub_download(repo_id=model_path, filename="config.json")
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        config = AMPLIFYConfig(**config_dict)
        self.plm = AMPLIFY(config)

        weight_file = hf_hub_download(repo_id=model_path, filename="model.safetensors")
        state_dict = safetensors.torch.load_file(weight_file)
        self.plm.load_state_dict(state_dict)
        
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
    ) -> MaskedLMOutput:
        if attention_mask is not None:
            pad_mask = _attention_mask_to_additive(attention_mask, self.plm.encoder.weight.dtype)
        else:
            pad_mask = None
        
        return self.plm(
            src=input_ids,
            pad_mask=pad_mask,
            output_attentions=output_attentions if output_attentions is not None else False,
            output_hidden_states=output_hidden_states,
        )


def get_amplify_tokenizer(preset: str):
    return AmplifyTokenizerWrapper(AutoTokenizer.from_pretrained(presets[preset], trust_remote_code=True))


def build_amplify_model(preset: str, masked_lm: bool = False) -> Tuple[nn.Module, AutoTokenizer]:
    model_path = presets[preset]
    if masked_lm:
        model = AmplifyForMaskedLM(model_path).eval()
    else:
        model = AmplifyForEmbedding(model_path).eval()
    tokenizer = get_amplify_tokenizer(preset)
    return model, tokenizer


def get_amplify_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    model_path = presets[preset]
    if hybrid:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()
    else:
        if tokenwise:
            model = AutoModelForTokenClassification.from_pretrained(
                model_path, num_labels=num_labels, trust_remote_code=True
            ).eval()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=num_labels, trust_remote_code=True
            ).eval()
    tokenizer = get_amplify_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.amplify
    model, tokenizer = build_amplify_model('AMPLIFY-120')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICLLLICIIVMLL'))
