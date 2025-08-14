from typing import Dict, List, Optional

import pandas as pd
import torch

from base_models.get_base_models import get_tokenizer
from ...embedder import Embedder, EmbeddingArguments
from .data_loader import load_proteingym_dms


def tokenize_df_sequences(df: pd.DataFrame, model_name: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Tokenize unique sequences present in a ProteinGym assay DataFrame using the
    selected model's tokenizer.

    Expects DataFrame columns: "mutated_seq", "target_seq".

    Returns a mapping: sequence -> { 'input_ids': Tensor, 'attention_mask': Tensor }
    """
    if not {'mutated_seq', 'target_seq'}.issubset(df.columns):
        raise KeyError("DataFrame must contain 'mutated_seq' and 'target_seq' columns")

    tokenizer = get_tokenizer(model_name)
    mutated = (df['mutated_seq'].tolist())
    targets = (df['target_seq'].tolist())
    unique_sequences = mutated + [t for t in targets if t not in mutated]
    # Batch tokenize for efficiency; align back to each sequence
    batch_tokens = tokenizer(
        unique_sequences,
        return_tensors='pt',
        padding='longest',
        pad_to_multiple_of=8,
        add_special_tokens=True,
    )
    tokens_by_sequence: Dict[str, Dict[str, torch.Tensor]] = {}
    for idx, seq in enumerate(unique_sequences):
        tokens_by_sequence[seq] = {
            'input_ids': batch_tokens['input_ids'][idx],
            'attention_mask': batch_tokens['attention_mask'][idx],
        }
    return tokens_by_sequence


def embed_df_sequences(
    df: pd.DataFrame,
    model_name: str,
    embedding_args: Optional[EmbeddingArguments] = None,
) -> Dict[str, torch.Tensor]:
    """
    Embed all unique sequences in a ProteinGym assay DataFrame using Protify's
    Embedder. Ensures that the wild-type (target) sequence is embedded only once
    per assay.

    Returns a dictionary: sequence -> embedding Tensor
    """
    if embedding_args is None:
        embedding_args = EmbeddingArguments(
            embedding_batch_size=8,
            embedding_num_workers=0,
            download_embeddings=False,
            matrix_embed=False,
            embedding_pooling_types=['mean'],
            save_embeddings=False,
            embed_dtype=torch.float32,
            sql=False,
            embedding_save_dir='embeddings',
        )

    if not {'mutated_seq', 'target_seq'}.issubset(df.columns):
        raise KeyError("DataFrame must contain 'mutated_seq' and 'target_seq' columns")

    mutated = (df['mutated_seq'].tolist())
    targets = (df['target_seq'].tolist())
    sequences_to_embed = mutated + [t for t in targets if t not in mutated]

    embedder = Embedder(embedding_args, sequences_to_embed)
    emb_dict = embedder(model_name) or {}
    if emb_dict is None:
        emb_dict = {}
    return emb_dict


def load_tokenize_and_embed_for_dms_ids(
    dms_ids: List[str],
    model_name: str,
    mode: Optional[str] = None,
    repo_id: str = "nikraf/ProteinGym_DMS",
    embedding_args: Optional[EmbeddingArguments] = None,
) -> Dict[str, Dict[str, object]]:
    """
    For each DMS_id:
      - Load its DataFrame
      - Tokenize sequences (mutants in order, then target once)
      - Embed those sequences with Protify's Embedder
    Returns mapping per DMS_id with the original df, tokens, and embeddings.
    """
    results: Dict[str, Dict[str, object]] = {}

    tokenizer = get_tokenizer(model_name)

    for dms_id in dms_ids:
        df = load_proteingym_dms(dms_id, mode=mode, repo_id=repo_id)
        if df is None or len(df) == 0:
            raise RuntimeError(f"No data loaded for DMS_id={dms_id}")

        mutated = (df['mutated_seq'].tolist())
        targets = (pd.unique(df['target_seq']).tolist())
        sequences = mutated + [t for t in targets if t not in mutated]

        # Tokenize per assay
        batch_tokens = tokenizer(
            sequences,
            return_tensors='pt',
            padding='longest',
            pad_to_multiple_of=8,
            add_special_tokens=True,
        )
        tokens_by_sequence: Dict[str, Dict[str, torch.Tensor]] = {}
        for idx, seq in enumerate(sequences):
            tokens_by_sequence[seq] = {
                'input_ids': batch_tokens['input_ids'][idx],
                'attention_mask': batch_tokens['attention_mask'][idx],
            }

        # Embed per assay
        embedder = Embedder(embedding_args, sequences)
        embedding_dict = embedder(model_name) or {}
        if embedding_dict is None:
            embedding_dict = {}

        results[dms_id] = {
            'df': df,
            'tokens_by_sequence': tokens_by_sequence,
            'embeddings': {seq: embedding_dict.get(seq) for seq in sequences if seq in embedding_dict},
        }

    return results
