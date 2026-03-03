import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple, Optional, Union, Sequence


class CharTokenizer:
    """
    Maps every symbol in `alphabet` to the *smallest* consecutive integers
    starting at 1.  Index 0 is reserved for padding, so the set of usable
    integers is exactly ``range(1, vocab_size)`` and is therefore the minimal
    range required to represent the alphabet.
    """
    pad_token_id: int = 0
    pad_token: str = "<pad>"
    cls_token_id: int = -1  # Will be set in __init__
    cls_token: str = "<cls>"
    eos_token_id: int = -2  # Will be set in __init__
    eos_token: str = "<eos>"

    def __init__(self, alphabet: str) -> None:
        self.alphabet: str = alphabet
        self.char2id: Dict[str, int] = {ch: i + 1 for i, ch in enumerate(alphabet)}
        self.id2char: Dict[int, str] = {i + 1: ch for i, ch in enumerate(alphabet)}
        
        # Add special tokens
        next_id = len(alphabet) + 1
        self.cls_token_id = next_id
        self.char2id[self.cls_token] = self.cls_token_id
        self.id2char[self.cls_token_id] = self.cls_token
        
        next_id += 1
        self.eos_token_id = next_id
        self.char2id[self.eos_token] = self.eos_token_id
        self.id2char[self.eos_token_id] = self.eos_token
        
        self.vocab_size: int = len(alphabet) + 3  # +3 for pad, cls, and eos

    def encode(self, seq: str) -> torch.Tensor:
        """Convert a single string to a *1-D LongTensor* of ids (no padding)."""
        try:
            # Add cls at the beginning and eos at the end
            ids = [self.cls_token_id] + [self.char2id[ch] for ch in seq] + [self.eos_token_id]
        except KeyError as e:  # unknown character
            raise ValueError(f"Unknown symbol {e} for this alphabet") from None
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Sequence[int], skip_pad: bool = True, skip_special: bool = True) -> str:
        """Convert a sequence of ids back to a string."""
        chars = []
        for idx in ids:
            # Skip special tokens if requested
            if (skip_pad and idx == self.pad_token_id) or \
               (skip_special and (idx == self.cls_token_id or idx == self.eos_token_id)):
                continue
            chars.append(self.id2char.get(int(idx), "?"))
        return "".join(chars)

    def __call__(
        self,
        sequences: Union[str, List[str], Tuple[str, ...]],
        return_tensors: str = "pt",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]

        # Tokenise each sequence
        encoded: List[torch.Tensor] = [self.encode(seq) for seq in sequences]

        # Pad on the **right** with `pad_token_id`
        input_ids: torch.Tensor = pad_sequence(
            encoded, batch_first=True, padding_value=self.pad_token_id
        )

        # Attention mask: 1 for real tokens, 0 for pad
        attention_mask = (input_ids != self.pad_token_id).to(torch.long)

        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        raise ValueError(f"Unsupported tensor type: {return_tensors}")


class OneHotModel(nn.Module):
    """
    Fast, parameter-free one-hot projection.

    Forward signature follows HuggingFace style:
        forward(input_ids, attention_mask=None) -> Tensor[B, L, vocab]
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        one_hot = F.one_hot(input_ids, num_classes=self.vocab_size).float()
        if attention_mask is not None:
            one_hot = one_hot * attention_mask.unsqueeze(-1)  # zero out pads
        return one_hot


AA_SET = 'LAGVSERTIPDKQNFYMHWCXBUOZ*'
CODON_SET = 'aA@bB#$%rRnNdDcCeEqQ^G&ghHiIj+MmlJLkK(fFpPoO=szZwSXTtxWyYuvUV]})'
DNA_SET = 'ATCG'
RNA_SET = 'AUCG'

ALPHABET_DICT = {
    'OneHot-Protein': AA_SET,
    'OneHot-DNA': DNA_SET,
    'OneHot-RNA': RNA_SET,
    'OneHot-Codon': CODON_SET
}


def build_one_hot_model(preset: str = 'OneHot-Protein', model_path: str = None, **kwargs):
    alphabet = str(ALPHABET_DICT[preset])
    tokenizer = CharTokenizer(alphabet)
    model = OneHotModel(tokenizer.vocab_size)
    return model, tokenizer


def get_one_hot_tokenizer(preset: str, model_path: str = None):
    return CharTokenizer(ALPHABET_DICT[preset])


if __name__ == '__main__':
    # py -m base_models.one_hot
    model, tokenizer = build_one_hot_model()          # default: protein

    sequences = ["ACGT", "PROTEIN"]
    batch = tokenizer(sequences)

    one_hot = model(**batch)                          # [2, 7, vocab]
    print(f"input_ids shape      : {batch['input_ids'].shape}")
    print(f"attention_mask shape : {batch['attention_mask'].shape}")
    print(f"one-hot shape        : {one_hot.shape}")

    print("\n--- first sequence, first 5 one-hot rows (trimmed) ---")
    print(one_hot[0, :5])

    # round-trip demo
    decoded = tokenizer.decode(batch["input_ids"][1])
    print("\nDecoded second sequence:", decoded)
    
    # Show tokens with special tokens
    print("\nInput with special tokens visible:")
    decoded_with_special = tokenizer.decode(batch["input_ids"][0], skip_special=False)
    print(decoded_with_special)
    
    # Show token IDs
    print("\nToken IDs for first sequence:")
    print(batch["input_ids"][0])
