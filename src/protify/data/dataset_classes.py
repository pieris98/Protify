### imports
import random
import torch
import numpy as np
import sqlite3
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from utils import print_message
from tqdm.auto import tqdm
from typing import List


class PairEmbedsLabelsDatasetFromDisk(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            col_a='SeqA',
            col_b='SeqB',
            label_col='labels',
            full=False, 
            db_path='embeddings.db',
            batch_size=64,
            read_scaler=100,
            input_size=768,
            task_type='regression',
            **kwargs
        ):
        self.seqs_a, self.seqs_b, self.labels = list(hf_dataset[col_a]), list(hf_dataset[col_b]), list(hf_dataset[label_col])
        self.db_file = db_path
        self.batch_size = batch_size
        self.input_size = input_size
        self.full = full
        self.length = len(self.labels)
        self.read_amt = read_scaler * self.batch_size
        self.embeddings_a, self.embeddings_b, self.current_labels = [], [], []
        self.count, self.index = 0, 0
        self.task_type = task_type

    def __len__(self):
        return self.length

    def check_seqs(self, all_seqs):
        missing_seqs = [seq for seq in self.seqs_a + self.seqs_b if seq not in all_seqs]
        if missing_seqs:
            print_message(f'Sequences not found in embeddings: {missing_seqs}')
        else:
            print_message('All sequences in embeddings')

    def reset_epoch(self):
        data = list(zip(self.seqs_a, self.seqs_b, self.labels))
        random.shuffle(data)
        self.seqs_a, self.seqs_b, self.labels = zip(*data)
        self.seqs_a, self.seqs_b, self.labels = list(self.seqs_a), list(self.seqs_b), list(self.labels)
        self.embeddings_a, self.embeddings_b, self.current_labels = [], [], []
        self.count, self.index = 0, 0

    def get_embedding(self, c, seq):
        result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (seq,))
        row = result.fetchone()
        if row is None:
            raise ValueError(f"Embedding not found for sequence: {seq}")
        emb_data = row[0]
        emb = torch.tensor(np.frombuffer(emb_data, dtype=np.float32).reshape(-1, self.input_size))
        return emb

    def read_embeddings(self):
        embeddings_a, embeddings_b, labels = [], [], []
        self.count += self.read_amt
        if self.count >= self.length:
            self.reset_epoch()
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        for i in range(self.count, self.count + self.read_amt):
            if i >= self.length:
                break
            emb_a = self.get_embedding(c, self.seqs_a[i])
            emb_b = self.get_embedding(c, self.seqs_b[i])
            embeddings_a.append(emb_a)
            embeddings_b.append(emb_b)
            labels.append(self.labels[i])
        conn.close()
        self.index = 0
        self.embeddings_a = embeddings_a
        self.embeddings_b = embeddings_b
        self.current_labels = labels

    def __getitem__(self, idx):
        if self.index >= len(self.current_labels) or len(self.current_labels) == 0:
            self.read_embeddings()

        emb_a = self.embeddings_a[self.index]
        emb_b = self.embeddings_b[self.index]
        label = self.current_labels[self.index]

        self.index += 1

        # 50% chance to switch the order of a and b
        if random.random() < 0.5:
            emb_a, emb_b = emb_b, emb_a

        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return emb_a, emb_b, label


class PairEmbedsLabelsDataset(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            emb_dict,
            col_a='SeqA',
            col_b='SeqB',
            full=False,
            label_col='labels',
            input_size=768,
            task_type='regression',
            **kwargs
        ):
        self.seqs_a = list(hf_dataset[col_a])
        self.seqs_b = list(hf_dataset[col_b])
        self.labels = list(hf_dataset[label_col])
        self.input_size = input_size // 2 if not full else input_size # already scaled if ppi
        self.task_type = task_type
        self.full = full

        # Combine seqs_a and seqs_b to find all unique sequences needed
        needed_seqs = set(list(hf_dataset[col_a]) + list(hf_dataset[col_b]))
        # Filter emb_dict to keep only the necessary embeddings
        self.emb_dict = {seq: emb_dict[seq] for seq in needed_seqs if seq in emb_dict}
        # Check for any missing embeddings
        missing_seqs = needed_seqs - self.emb_dict.keys()
        if missing_seqs:
            raise ValueError(f"Embeddings not found for sequences: {missing_seqs}")

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        seq_a, seq_b = self.seqs_a[idx], self.seqs_b[idx]
        emb_a = self.emb_dict.get(seq_a).reshape(-1, self.input_size)
        emb_b = self.emb_dict.get(seq_b).reshape(-1, self.input_size)
        
        # 50% chance to switch the order of a and b
        if random.random() < 0.5:
            emb_a, emb_b = emb_b, emb_a

        # Prepare the label
        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)

        return emb_a, emb_b, label


class EmbedsLabelsDatasetFromDisk(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            col_name='seqs',
            label_col='labels',
            full=False,
            db_path='embeddings.db',
            batch_size=64,
            read_scaler=100,
            input_size=768,
            task_type='singlelabel',
            **kwargs
        ): 
        self.seqs, self.labels = list(hf_dataset[col_name]), list(hf_dataset[label_col])
        self.length = len(self.labels)
        self.max_length = len(max(self.seqs, key=len))
        print_message(f'Max length: {self.max_length}')

        self.db_file = db_path
        self.batch_size = batch_size
        self.input_size = input_size
        self.full = full

        self.task_type = task_type
        self.read_amt = read_scaler * self.batch_size
        self.embeddings, self.current_labels = [], []
        self.count, self.index = 0, 0

        self.reset_epoch()

    def __len__(self):
        return self.length

    def check_seqs(self, all_seqs):
        cond = False
        for seq in self.seqs:
            if seq not in all_seqs:
                cond = True
            if cond:
                break
        if cond:
            print_message('Sequences not found in embeddings')
        else:
            print_message('All sequences in embeddings')

    def reset_epoch(self):
        data = list(zip(self.seqs, self.labels))
        random.shuffle(data)
        self.seqs, self.labels = zip(*data)
        self.seqs, self.labels = list(self.seqs), list(self.labels)
        self.embeddings, self.current_labels = [], []
        self.count, self.index = 0, 0

    def read_embeddings(self):
        embeddings, labels = [], []
        self.count += self.read_amt
        if self.count >= self.length:
            self.reset_epoch()
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        for i in range(self.count, self.count + self.read_amt):
            if i >= self.length:
                break
            result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (self.seqs[i],))
            row = result.fetchone()
            emb_data = row[0]
            emb = torch.tensor(np.frombuffer(emb_data, dtype=np.float32).reshape(-1, self.input_size))
            if self.full:
                padding_needed = self.max_length - emb.size(0)
                emb = F.pad(emb, (0, 0, 0, padding_needed), value=0)
            embeddings.append(emb)
            labels.append(self.labels[i])
        conn.close()
        self.index = 0
        self.embeddings = embeddings
        self.current_labels = labels

    def __getitem__(self, idx):
        if self.index >= len(self.current_labels) or len(self.current_labels) == 0:
            self.read_embeddings()

        emb = self.embeddings[self.index]
        label = self.current_labels[self.index]

        self.index += 1

        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return emb.squeeze(0), label


class EmbedsLabelsDataset(TorchDataset):
    def __init__(self, hf_dataset, emb_dict, col_name='seqs', label_col='labels', task_type='singlelabel', full=False, **kwargs):
        self.embeddings = self.get_embs(emb_dict, list(hf_dataset[col_name]))
        self.full = full
        self.labels = list(hf_dataset[label_col])
        self.task_type = task_type
        self.max_length = len(max(list(hf_dataset[col_name]), key=len))
        print_message(f'Max length: {self.max_length}')

    def __len__(self):
        return len(self.labels)
    
    def get_embs(self, emb_dict, seqs):
        embeddings = []
        for seq in tqdm(seqs, desc='Loading Embeddings'):
            emb = emb_dict[seq]
            embeddings.append(emb)
        return embeddings

    def __getitem__(self, idx):
        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        emb = self.embeddings[idx].float()
        if self.full:
            padding_needed = self.max_length - emb.size(0)
            emb = F.pad(emb, (0, 0, 0, padding_needed), value=0)
        return emb.squeeze(0), label
    

class StringLabelDataset(TorchDataset):    
    def __init__(self, hf_dataset, col_name='seqs', label_col='labels', **kwargs):
        self.seqs = list(hf_dataset[col_name])
        self.labels = list(hf_dataset[label_col])
        self.lengths = [len(seq) for seq in self.seqs]

    def avg(self):
        return sum(self.lengths) / len(self.lengths)

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        return seq, label
    

class PairStringLabelDataset(TorchDataset):
    def __init__(self, hf_dataset, col_a='SeqA', col_b='SeqB', label_col='labels', train=True, **kwargs):
        self.seqs_a, self.seqs_b = list(hf_dataset[col_a]), list(hf_dataset[col_b])
        self.labels = list(hf_dataset[label_col])
        self.train = train

    def avg(self):
        return sum(len(seqa) + len(seqb) for seqa, seqb in zip(self.seqs_a, self.seqs_b)) / len(self.seqs_a)

    def __len__(self):
        return len(self.seqs_a)

    def __getitem__(self, idx):
        seq_a, seq_b = self.seqs_a[idx], self.seqs_b[idx]
        if self.train and random.random() < 0.5:
            seq_a, seq_b = seq_b, seq_a
        return seq_a, seq_b, self.labels[idx]


class SimpleProteinDataset(TorchDataset):
    """Simple dataset for protein sequences."""
    def __init__(self, sequences: List[str]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


class MultiEmbedsLabelsDatasetFromDisk(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            seq_cols: List[str],
            label_col: str = 'labels',
            full: bool = False,
            db_path: str = 'embeddings.db',
            batch_size: int = 64,
            read_scaler: int = 100,
            input_size: int = 768,
            task_type: str = 'singlelabel',
            train: bool = True,
            **kwargs,
        ):
        self.seq_cols = seq_cols
        self.labels = list(hf_dataset[label_col])
        self.length = len(self.labels)
        self.full = full
        self.db_file = db_path
        self.batch_size = batch_size
        self.read_amt = read_scaler * self.batch_size
        self.input_size = input_size // len(seq_cols) if not full else input_size # already scaled if multi-column
        self.task_type = task_type
        self.train = train

        # Store sequences per column
        self.col_to_seqs = {col: list(hf_dataset[col]) for col in seq_cols}

        # Precompute max combined length for matrix embeddings from raw strings
        if self.full:
            def combined_len_at(i: int) -> int:
                return sum(len(self.col_to_seqs[c][i]) for c in self.seq_cols) + (len(self.seq_cols) - 1)
            self.max_length = max(combined_len_at(i) for i in range(self.length)) if self.length > 0 else 0

        self.embeddings, self.current_labels = [], []
        self.count, self.index = 0, 0

    def __len__(self):
        return self.length

    def reset_epoch(self):
        # shuffle consistently across columns
        idxs = list(range(self.length))
        random.shuffle(idxs)
        for col in self.seq_cols:
            self.col_to_seqs[col] = [self.col_to_seqs[col][i] for i in idxs]
        self.labels = [self.labels[i] for i in idxs]
        self.embeddings, self.current_labels = [], []
        self.count, self.index = 0, 0

    def _get_embedding(self, c, seq: str) -> torch.Tensor:
        result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (seq,))
        row = result.fetchone()
        if row is None:
            raise ValueError(f"Embedding not found for sequence: {seq}")
        emb_data = row[0]
        emb = torch.tensor(np.frombuffer(emb_data, dtype=np.float32).reshape(-1, self.input_size))
        return emb

    def _combine_matrix(self, parts: List[torch.Tensor]) -> torch.Tensor:
        # Insert a single zero row between parts
        if len(parts) == 0:
            return torch.zeros(0, self.input_size)
        sep = torch.zeros(1, self.input_size, dtype=parts[0].dtype)
        out = []
        for i, p in enumerate(parts):
            out.append(p)
            if i < len(parts) - 1:
                out.append(sep)
        return torch.cat(out, dim=0)

    def read_embeddings(self):
        embeddings, labels = [], []
        self.count += self.read_amt
        if self.count >= self.length:
            self.reset_epoch()
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        for i in range(self.count, self.count + self.read_amt):
            if i >= self.length:
                break
            parts = [self._get_embedding(c, self.col_to_seqs[col][i]) for col in self.seq_cols]
            if self.full:
                emb = self._combine_matrix(parts)
                # pad to max_length
                if self.full and self.max_length:
                    pad_needed = self.max_length - emb.size(0)
                    if pad_needed > 0:
                        emb = F.pad(emb, (0, 0, 0, pad_needed), value=0)
            else:
                # vector embeddings are 1 x d; concatenate along feature dim
                emb = torch.cat([p.reshape(1, -1) for p in parts], dim=-1)
            embeddings.append(emb)
            labels.append(self.labels[i])
        conn.close()
        self.index = 0
        self.embeddings = embeddings
        self.current_labels = labels

    def __getitem__(self, idx):
        if self.index >= len(self.current_labels) or len(self.current_labels) == 0:
            self.read_embeddings()

        emb = self.embeddings[self.index]
        label = self.current_labels[self.index]
        self.index += 1

        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return emb.squeeze(0), label


class MultiEmbedsLabelsDataset(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            seq_cols: List[str],
            label_col: str = 'labels',
            full: bool = False,
            emb_dict: dict = None,
            input_size: int = 768,
            task_type: str = 'singlelabel',
            train: bool = True,
            **kwargs,
        ):
        self.seq_cols = seq_cols
        self.labels = list(hf_dataset[label_col])
        self.full = full
        self.input_size = input_size // len(seq_cols) if not full else input_size
        self.task_type = task_type
        self.train = train

        self.col_to_seqs = {col: list(hf_dataset[col]) for col in seq_cols}

        # Precompute combined embeddings
        self.embeddings = []
        if self.full:
            # compute max_length from strings
            def combined_len_at(i: int) -> int:
                return sum(len(self.col_to_seqs[c][i]) for c in self.seq_cols) + (len(self.seq_cols) - 1)
            self.max_length = max(combined_len_at(i) for i in range(len(self.labels))) if len(self.labels) > 0 else 0

        for i in tqdm(range(len(self.labels)), desc='Loading Multi-Embeddings'):
            parts = []
            for col in self.seq_cols:
                seq = self.col_to_seqs[col][i]
                emb = emb_dict[seq]
                emb = emb.reshape(-1, self.input_size)
                parts.append(emb)
            if self.full:
                emb = self._combine_matrix(parts)
                # pad to max_length
                if self.max_length:
                    pad_needed = self.max_length - emb.size(0)
                    if pad_needed > 0:
                        emb = F.pad(emb, (0, 0, 0, pad_needed), value=0)
            else:
                emb = torch.cat([p.reshape(1, -1) for p in parts], dim=-1)
            self.embeddings.append(emb)

    def _combine_matrix(self, parts: List[torch.Tensor]) -> torch.Tensor:
        if len(parts) == 0:
            return torch.zeros(0, self.input_size)
        sep = torch.zeros(1, self.input_size, dtype=parts[0].dtype)
        out = []
        for i, p in enumerate(parts):
            out.append(p)
            if i < len(parts) - 1:
                out.append(sep)
        return torch.cat(out, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.task_type in ['multilabel', 'regression', 'sigmoid_regression']:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        emb = self.embeddings[idx].float()
        return emb.squeeze(0), label
    