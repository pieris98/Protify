import os
import pandas as pd
from typing import Optional, Tuple, Dict, List
import torch

from .data_loader import load_proteingym_dms
from embedder import EmbeddingArguments, Embedder
from utils import torch_load
from .scoring_utils import get_sequence_slices, _infer_model_context_len


def prepare_supervised_dms_for_probe(
    dms_id: str,
    model_name: str,
    *,
    mode: str = 'supervised',
    repo_id: str = 'GleghornLab/ProteinGym_DMS',
    hf_token: Optional[str] = None,
    scoring_window: str = 'optimal',
    embedding_args: Optional[EmbeddingArguments] = None,
) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Prepare supervised DMS data for modeling.

    Returns:
      - df: the original ProteinGym dataframe (for writing predictions later)
      - embeddings_dict: mapping sequence -> embedding tensor
      - labels: torch float tensor of DMS scores
    """

    # Load the ProteinGym DMS assay rows
    df_original = load_proteingym_dms(
        dms_id=dms_id,
        mode=mode,
        repo_id=repo_id,
        hf_token=hf_token,
    )

    df_sliced = get_sequence_slices(
        df=df_original.copy(),
        target_seq=df_original['target_seq'].iloc[0],
        model_context_len=_infer_model_context_len(model_name, tokenizer),
        start_idx=1,
        scoring_window=scoring_window,
        indel_mode=False
    )

    # Build the embeddings dict for unique mutated sequences
    sliced_sequences = df_sliced['sliced_mutated_seq'].astype(str).tolist()
    unique_sequences = list(dict.fromkeys(sliced_sequences))

    if embedding_args is None:
        embedding_args = EmbeddingArguments()
        embedding_args.save_embeddings = False

    embedder = Embedder(embedding_args, unique_sequences)
    embeddings_dict = embedder(model_name)

    # Fallback: if nothing new was embedded, try loading from disk if present
    if embeddings_dict is None:
        save_path = os.path.join(
            embedding_args.embedding_save_dir,
            f"{model_name}_{embedding_args.matrix_embed}.pth",
        )
        if os.path.exists(save_path):
            embeddings_dict = torch_load(save_path)
        else:
            embeddings_dict = {}

    # Exclude the wild-type rows introduced during slicing
    df_mutants = df_sliced[df_sliced['mutated_seq'] != df_sliced['target_seq'].iloc[0]]
    # Labels tensor from DMS_score
    labels = torch.tensor(df_mutants['DMS_score'].astype(float).values, dtype=torch.float32)

    return df_mutants, embeddings_dict, labels



'''
def prepare_supervised_dms_for_transformer_probe(
    dms_id: str,
    model_name: str,
    *,
    mode: str = 'supervised',
    repo_id: str = 'GleghornLab/ProteinGym_DMS',
    hf_token: Optional[str] = None,
    scoring_window: str = 'optimal',
    selected_mode: str = 'supervised',
) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor], torch.Tensor]:
    
    #TODO: Implement this
'''


def get_cv_fold_variables(selected_mode: str) -> List[str]:
    """
    Return the list of fold column names to use for a given ProteinGym supervised mode.
    """
    if selected_mode == 'indels_supervised':
        return [
            'fold_random_5_indels'
        ]
    elif selected_mode == 'supervised_multiples':
        return [
            'fold_rand_multiples'
        ]
    # default: standard single-mutant supervised
    return [
        'fold_random_5',
        'fold_modulo_5',
        'fold_contiguous_5',
    ]
