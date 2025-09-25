import torch
import torch.nn as nn
from typing import Optional
from transformers import EsmTokenizer, EsmConfig
from model_components.transformer import TransformerForMaskedLM, TransformerConfig


presets = {
    'Random': 'random',
    'Random-Transformer': 'facebook/esm2_t12_35M_UR50D', # default is 35M version
    'Random-ESM2-8': 'facebook/esm2_t6_8M_UR50D',
    'Random-ESM2-35': 'facebook/esm2_t12_35M_UR50D',
    'Random-ESM2-150': 'facebook/esm2_t30_150M_UR50D',
    'Random-ESM2-650': 'facebook/esm2_t36_650M_UR50D',
}


class RandomModel(nn.Module):
    def __init__(self, config: EsmConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.holder_param = torch.nn.Parameter(torch.randn(1, 1, self.hidden_size))
        # Simple projection head to produce token logits
        self.lm_head = nn.Linear(self.hidden_size, config.vocab_size)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        device = self.holder_param.device
        last_hidden_state = torch.randn(input_ids.shape[0], input_ids.shape[1], self.hidden_size, device=device)
        logits = self.lm_head(last_hidden_state)
        return {"last_hidden_state": last_hidden_state, "logits": logits}


class RandomTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.transformer = TransformerForMaskedLM(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> torch.Tensor:
        if output_attentions:
            out = self.transformer(input_ids, attention_mask, output_attentions=output_attentions)
            return out.last_hidden_state, out.attentions
        else:
            return self.transformer(input_ids, attention_mask).last_hidden_state


def build_random_model(preset: str):
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    if preset == 'Random':
        model = RandomModel(EsmConfig.from_pretrained('facebook/esm2_t12_35M_UR50D'))
    else:
        esm_config = EsmConfig.from_pretrained(presets[preset])
        config = TransformerConfig()
        config.hidden_size = esm_config.hidden_size
        config.n_heads = esm_config.num_attention_heads
        config.n_layers = esm_config.num_hidden_layers
        config.vocab_size = esm_config.vocab_size
        model = RandomTransformer(config).eval()
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = build_random_model('Random-Transformer')
    print(model)
    print(tokenizer)