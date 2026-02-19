"""
We use the FastESM2 implementation of ESM2.
"""
import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    EsmTokenizer,
)
from .base_tokenizer import BaseSequenceTokenizer


presets = {
    'ESM2-8': 'Synthyra/ESM2-8M',
    'ESM2-35': 'Synthyra/ESM2-35M',
    'ESM2-150': 'Synthyra/ESM2-150M',
    'ESM2-650': 'Synthyra/ESM2-650M',
    'ESM2-3B': 'Synthyra/ESM2-3B',
    'DSM-150': 'GleghornLab/DSM_150',
    'DSM-650': 'GleghornLab/DSM_650',
    'DSM-PPI': 'Synthyra/DSM_ppi_full',
}


class ESM2TokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: EsmTokenizer):
        super().__init__(tokenizer)

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'longest')
        kwargs.setdefault('add_special_tokens', True)
        tokenized = self.tokenizer(sequences, **kwargs)
        return tokenized


class FastEsmForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.esm = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = False,
            **kwargs,
    ) -> torch.Tensor:
        if output_attentions:
            out = self.esm(input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
            return out.last_hidden_state, out.attentions
        else:
            return self.esm(input_ids, attention_mask=attention_mask).last_hidden_state


def get_esm2_tokenizer(preset: str):
    return ESM2TokenizerWrapper(EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'))


def build_esm2_model(preset: str, masked_lm: bool = False, **kwargs):
    if masked_lm:
        model = AutoModelForMaskedLM.from_pretrained(presets[preset], trust_remote_code=True).eval()
    else:
        model = FastEsmForEmbedding(presets[preset]).eval()
    tokenizer = get_esm2_tokenizer(preset)
    return model, tokenizer


def get_esm2_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    model_path = presets[preset]
    if hybrid:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()
    else:
        if tokenwise:
            model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_labels, trust_remote_code=True).eval()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, trust_remote_code=True).eval()
    tokenizer = get_esm2_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.esm2
    model, tokenizer = build_esm2_model('ESM2-8')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
