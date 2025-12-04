import torch
import torch.nn as nn
import re
from typing import Optional, Union, List, Dict
from transformers import T5EncoderModel, T5Tokenizer

from .t5 import T5ForSequenceClassification, T5ForTokenClassification
from .base_tokenizer import BaseSequenceTokenizer


presets = {
    'ProtT5': 'Rostlab/prot_t5_xl_half_uniref50-enc',
    'ProtT5-XL-UniRef50-full-prec': 'Rostlab/prot_t5_xl_uniref50',
    'ProtT5-XXL-UniRef50': 'Rostlab/prot_t5_xxl_uniref50',
    'ProtT5-XL-BFD': 'Rostlab/prot_t5_xl_bfd',
    'ProtT5-XXL-BFD': 'Rostlab/prot_t5_xxl_bfd',
}


class T5TokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: T5Tokenizer):
        super().__init__(tokenizer)

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'longest')
        kwargs.setdefault('add_special_tokens', True)
        sequences = [re.sub(r"[UZOB]", "X", seq) for seq in sequences]
        sequences = [' '.join(seq) for seq in sequences]
        tokenized = self.tokenizer(sequences, **kwargs)
        return tokenized


class Prott5ForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.plm = T5EncoderModel.from_pretrained(model_path)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = False,
            **kwargs,
    ) -> torch.Tensor:
        if output_attentions:
            out = self.plm(input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
            return out.last_hidden_state, out.attentions
        else:
            return self.plm(input_ids, attention_mask=attention_mask).last_hidden_state


def get_prott5_tokenizer(preset: str):
    return T5TokenizerWrapper(T5Tokenizer.from_pretrained(presets[preset]))


def build_prott5_model(preset: str, masked_lm: bool = False, **kwargs):
    model_path = presets[preset]
    model = Prott5ForEmbedding(model_path).eval()
    tokenizer = get_prott5_tokenizer(preset)
    return model, tokenizer


def get_prott5_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    model_path = presets[preset]
    if hybrid:
        model = T5EncoderModel.from_pretrained(model_path).eval()
    else:
        if tokenwise:
            model = T5ForTokenClassification.from_pretrained(model_path, num_labels=num_labels).eval()
        else:
            model = T5ForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).eval()
    tokenizer = get_prott5_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.prott5
    model, tokenizer = build_prott5_model('ProtT5')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
