import torch
from typing import List, Tuple, Dict, Union
from .utils import pad_and_concatenate_dimer


def _pad_matrix_embeds(embeds: List[torch.Tensor], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # pad and concatenate, return padded embeds and mask
    padded_embeds, attention_masks = [], []
    for embed in embeds:
        seq_len = embed.size(0)
        padding_size = max_len - seq_len
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(max_len, dtype=torch.long)
        if padding_size > 0:
            attention_mask[seq_len:] = 0
            
            # Pad along the sequence dimension (dim=0)
            padding = torch.zeros((padding_size, embed.size(1)), dtype=embed.dtype)
            padded_embed = torch.cat((embed, padding), dim=0)
        else:
            padded_embed = embed
            
        padded_embeds.append(padded_embed)
        attention_masks.append(attention_mask)
        
    return torch.stack(padded_embeds), torch.stack(attention_masks)


class StringCollator:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

    def __call__(self, batch: Tuple[List[str], List[str]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer(batch,
                          padding='longest',
                          return_tensors='pt',
                          add_special_tokens=True)
        return batch


class StringLabelsCollator:
    def __init__(self, tokenizer, task_type='tokenwise', **kwargs):
        self.tokenizer = tokenizer
        self.task_type = task_type

    def __call__(self, batch: List[Tuple[str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs = [ex[0] for ex in batch]
        labels = [ex[1] for ex in batch]

        # Tokenize the sequences
        batch_encoding = self.tokenizer(
            seqs,
            padding='longest',
            truncation=False,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        # Handle labels based on tokenwise flag
        if self.task_type == 'tokenwise':
            # For token-wise labels, we need to pad to match the tokenized sequence length
            attention_mask = batch_encoding['attention_mask']
            lengths = [torch.sum(attention_mask[i]).item() for i in range(len(batch))]
            max_length = max(lengths)

            padded_labels = []
            for label in labels:
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label)

                label = label.flatten()
                padding_size = max_length - len(label)
                # Pad or truncate labels to match tokenized sequence length
                if padding_size > 0:
                    # Pad with -100 (ignored by loss functions)
                    padding = torch.full((padding_size,), -100, dtype=label.dtype)
                    padded_label = torch.cat((label, padding))
                else:
                    padded_label = label[:max_length]
                padded_labels.append(padded_label)
            
            # Stack all padded labels
            batch_encoding['labels'] = torch.stack(padded_labels)
        else:
            # For sequence-level labels, just stack them
            batch_encoding['labels'] = torch.stack([torch.tensor(ex[1]) for ex in batch])

        if self.task_type == 'multilabel':
            batch_encoding['labels'] = batch_encoding['labels'].float()
        else:
            batch_encoding['labels'] = batch_encoding['labels'].long()
        
        return batch_encoding


class EmbedsLabelsCollator:
    def __init__(self, full=False, task_type='tokenwise', **kwargs):
        self.full = full
        self.task_type = task_type
        
    def __call__(self, batch: List[Tuple[torch.Tensor, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        if self.full:
            embeds = [ex[0] for ex in batch]
            labels = [ex[1] for ex in batch]
            
            # Find max sequence length for padding
            max_length = max(embed.size(0) for embed in embeds)
            
            embeds, attention_mask = _pad_matrix_embeds(embeds, max_length)
            
            # Pad labels
            print('labels')
            if self.task_type == 'tokenwise':
                padded_labels = []
                for label in labels:
                    if not isinstance(label, torch.Tensor):
                        label = torch.tensor(label)

                    label = label.flatten()
                    padding_size = max_length - len(label)
                    if padding_size > 0:
                        # Use -100 as padding value for labels (ignored by loss functions)
                        padding = torch.full((padding_size,), -100, dtype=label.dtype)
                        padded_label = torch.cat((label, padding))
                    else:
                        padded_label = label[:max_length]
                    padded_labels.append(padded_label)
                    print(padded_label.shape)
            else:
                padded_labels = labels
            
            print('embeds')
            for embed in embeds:
                print(embed.shape)
                
            labels = torch.stack(padded_labels)

            if self.task_type == 'multilabel':
                labels = labels.float()
            else:
                labels = labels.long()

            return {
                'embeddings': embeds,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        else:
            embeds = torch.stack([ex[0] for ex in batch])
            labels = torch.stack([ex[1] for ex in batch])

            if self.task_type == 'multilabel':
                labels = labels.float()
            else:
                labels = labels.long()

            return {
                'embeddings': embeds,
                'labels': labels
            }


class PairCollator_input_ids:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Tuple[str, str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs_a, seqs_b, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.float)
        tokenized = self.tokenizer(
            seqs_a, seqs_b,
            padding='longest',
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }


class PairCollator_ab:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Tuple[str, str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs_a, seqs_b, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.float)
        tokenized_a = self.tokenizer(
            seqs_a,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        tokenized_b = self.tokenizer(
            seqs_b,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids_a': tokenized_a['input_ids'],
            'input_ids_b': tokenized_b['input_ids'],
            'attention_mask_a': tokenized_a['attention_mask'],
            'attention_mask_b': tokenized_b['attention_mask'],
            'labels': labels
        }


class PairEmbedsLabelsCollator:
    def __init__(self, full=False, **kwargs):
        self.full = full
        
    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        if self.full:
            embeds_a = [ex[0] for ex in batch]
            embeds_b = [ex[1] for ex in batch]
            max_len_a = max(embed.size(0) for embed in embeds_a)
            max_len_b = max(embed.size(0) for embed in embeds_b)
            embeds_a, attention_mask_a = _pad_matrix_embeds(embeds_a, max_len_a)
            embeds_b, attention_mask_b = _pad_matrix_embeds(embeds_b, max_len_b)
            embeds, attention_mask = pad_and_concatenate_dimer(embeds_a, embeds_b, attention_mask_a, attention_mask_b)

            labels = torch.stack([ex[2] for ex in batch])

            return {
                'embeddings': embeds,
                'attention_mask': attention_mask,
                'labels': labels
            }
        else:
            embeds_a = torch.stack([ex[0] for ex in batch])
            embeds_b = torch.stack([ex[1] for ex in batch]) 
            labels = torch.stack([ex[2] for ex in batch])
            embeds = torch.cat([embeds_a, embeds_b], dim=-1)
            return {
                'embeddings': embeds,
                'labels': labels
            }


class OneHotCollator:
    def __init__(self, alphabet="ACDEFGHIKLMNPQRSTVWY"):
        # Add X for unknown amino acids, and special CLS and EOS tokens
        alphabet = alphabet + "X"
        alphabet = list(alphabet)
        self.mapping = {token: idx for idx, token in enumerate(alphabet)}
        
    def __call__(self, batch):
        seqs = [ex[0] for ex in batch]
        labels = torch.stack([torch.tensor(ex[1]) for ex in batch])
        
        # Find the longest sequence in the batch
        max_len = max(len(seq) for seq in seqs)
        
        # One-hot encode and pad each sequence
        one_hot_tensors, attention_masks = [], []
        
        for seq in seqs:
            seq = list(seq)
            # Create one-hot encoding for each sequence (including CLS and EOS)
            seq_len = len(seq)
            one_hot = torch.zeros(seq_len, len(self.alphabet))
            
            # Add sequence tokens in the middle
            for pos, token in enumerate(seq):
                if token in self.mapping:
                    one_hot[pos, self.mapping[token]] = 1.0
                else:
                    # For non-canonical amino acids, use the X token
                    one_hot[pos, self.mapping["X"]] = 1.0
            
            # Create attention mask (1 for actual tokens, 0 for padding)
            attention_mask = torch.ones(seq_len)
            
            # Pad to the max length in this batch
            padding_size = max_len - seq_len
            if padding_size > 0:
                padding = torch.zeros(padding_size, len(self.alphabet))
                one_hot = torch.cat([one_hot, padding], dim=0)
                # Add zeros to attention mask for padding
                mask_padding = torch.zeros(padding_size)
                attention_mask = torch.cat([attention_mask, mask_padding], dim=0)
            
            one_hot_tensors.append(one_hot)
            attention_masks.append(attention_mask)
        
        # Stack all tensors in the batch
        embeddings = torch.stack(one_hot_tensors)
        attention_masks = torch.stack(attention_masks)
        
        return {
            'embeddings': embeddings,
            'attention_mask': attention_masks,
            'labels': labels,
        }
