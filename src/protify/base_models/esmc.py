"""
We use the FastPLM implementation of ESMC.
"""
import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
)
from .base_tokenizer import BaseSequenceTokenizer
from .esmc_utils import EsmSequenceTokenizer


presets = {
    'ESMC-300': 'Synthyra/ESMplusplus_small',
    'ESMC-600': 'Synthyra/ESMplusplus_large',
}


class ESMTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: EsmSequenceTokenizer):
        super().__init__(tokenizer)

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'longest')
        kwargs.setdefault('add_special_tokens', True)
        tokenized = self.tokenizer(sequences, **kwargs)
        return tokenized


class ESMplusplusForEmbedding(nn.Module):
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
            out = self.esm(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
            return out.last_hidden_state, out.attentions
        else:
            return self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def get_esmc_tokenizer(preset: str):
    tokenizer = EsmSequenceTokenizer()
    return ESMTokenizerWrapper(tokenizer)


def build_esmc_model(preset: str, masked_lm: bool = False, **kwargs):
    if masked_lm:
        model = AutoModelForMaskedLM.from_pretrained(presets[preset], trust_remote_code=True).eval()
    else:
        model = ESMplusplusForEmbedding(presets[preset]).eval()
    tokenizer = get_esmc_tokenizer(preset)
    return model, tokenizer


def get_esmc_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    model_path = presets[preset]
    if hybrid:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()
    else:
        if tokenwise:
            model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_labels, trust_remote_code=True).eval()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, trust_remote_code=True).eval()
    tokenizer = get_esmc_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.esmc
    model, tokenizer = build_esmc_model('ESMC-300')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
