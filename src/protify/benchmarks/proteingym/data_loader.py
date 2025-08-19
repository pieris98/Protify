import os
import json
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def _load_parquet_by_dms(repo_id: str, dms_id: str, hf_token: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Try to load a single-assay parquet shard from the Hub at by_dms_id/{sanitized}.parquet.
    Returns a DataFrame if found; otherwise None.
    """
    candidates = []
    id_str = str(dms_id)
    # Plain filename using the provided id
    candidates.append(f"by_dms_id/{id_str}.parquet")
    # Hashed variant used by repack script
    short_hash = hashlib.sha1(str(dms_id).encode("utf-8")).hexdigest()[:8]
    candidates.append(f"by_dms_id/{id_str}__{short_hash}.parquet")
    for filename in candidates:
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=hf_token)
        except Exception:
            continue
        try:
            df = pd.read_parquet(local_path)
            df = df[["DMS_id","mutated_seq","target_seq","DMS_score","DMS_score_bin","mutant"]] 
            return df
        except Exception:
            continue
    return None


def _load_via_index_select(repo_id: str, dms_id: str, hf_token: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Use an index.json mapping DMS_id -> row indices to select rows efficiently
    from the cached Arrow dataset without a full scan.
    Returns DataFrame if the index exists and is usable; otherwise None.
    """
    try:
        index_path = hf_hub_download(repo_id=repo_id, filename="index.json", repo_type="dataset", token=hf_token)
    except Exception:
        return None
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
    except Exception:
        return None
    indices = index.get(str(dms_id))
    if not indices:
        return None
    base = load_dataset(repo_id, split="train", streaming=False, token=hf_token)
    try:
        subset = base.select(indices)
        df = subset.to_pandas().reset_index(drop=True)
        df = df[["DMS_id","mutated_seq","target_seq","DMS_score","DMS_score_bin","mutant"]]
        return df
    except Exception:
        return None


def load_proteingym_dms(dms_id: str, mode: Optional[str] = None, repo_id: str = "GleghornLab/ProteinGym_DMS", hf_token: Optional[str] = None) -> pd.DataFrame:
    """
    Loads a single ProteinGym DMS assay from Hugging Face.
    """
    # per-assay parquet shard
    df = _load_parquet_by_dms(repo_id=repo_id, dms_id=dms_id, hf_token=hf_token)
    if df is None:
        # use precomputed index to select
        df = _load_via_index_select(repo_id=repo_id, dms_id=dms_id, hf_token=hf_token)
    if df is None:
        # try streaming filter to avoid materializing full dataset
        try:
            hf_stream = load_dataset(repo_id, split="train", streaming=True, token=hf_token)
            rows = [row for row in hf_stream if row.get("DMS_id", None) == dms_id]
            df = pd.DataFrame.from_records(rows)
            df = df[["DMS_id","mutated_seq","target_seq","DMS_score","DMS_score_bin","mutant"]] 
        except Exception:
            df = None
    if df is None or len(df) == 0:
        raise ValueError(f"No data loaded for DMS_id={dms_id}")
    
    if mode == 'benchmark':
        if 'is_indel' in df.columns:
            df = df[df['is_indel'] == False]
    elif mode == 'singles':
        if 'num_mutations' in df.columns:
            df = df[df['num_mutations'] == 1]
    elif mode == 'indels':
        if 'is_indel' in df.columns:
            df = df[df['is_indel'] == True]
    elif mode == 'multiple':
        if 'num_mutations' in df.columns:
            df = df[df['num_mutations'] > 1]

    return df.reset_index(drop=True)



class ProteinGymCVSplitter:
    @staticmethod
    def random_split(n_items: int, n_folds: int = 5, seed: int = 42) -> List[Dict[str, np.ndarray]]:
        """Standard random k-fold CV returning list of fold index dicts."""
        rng = np.random.default_rng(seed)
        indices = np.arange(n_items)
        rng.shuffle(indices)
        folds = np.array_split(indices, n_folds)
        splits = []
        for i in range(n_folds):
            test_idx = folds[i]
            valid_idx = folds[(i + 1) % n_folds]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j not in [i, (i + 1) % n_folds]])
            splits.append({
                'train_idx': train_idx,
                'valid_idx': valid_idx,
                'test_idx': test_idx,
            })
        return splits


    @staticmethod
    def contiguous_split(n_items: int, n_folds: int = 5) -> List[Dict[str, np.ndarray]]:
        """Contiguous block k-folds by row order. Next block is validation."""
        indices = np.arange(n_items)
        folds = np.array_split(indices, n_folds)
        splits = []
        for i in range(n_folds):
            test_idx = folds[i]
            valid_idx = folds[(i + 1) % n_folds]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j not in [i, (i + 1) % n_folds]])
            splits.append({
                'train_idx': train_idx,
                'valid_idx': valid_idx,
                'test_idx': test_idx,
            })
        return splits

    @staticmethod
    def modulo_split(n_items: int, n_folds: int = 5) -> List[Dict[str, np.ndarray]]:
        """Deterministic k-fold by index modulo. Fold i = {idx | idx % n_folds == i}."""
        indices = np.arange(n_items)
        splits = []
        for i in range(n_folds):
            test_idx = indices[indices % n_folds == i]
            valid_fold = (i + 1) % n_folds
            valid_idx = indices[indices % n_folds == valid_fold]
            train_idx = indices[(indices % n_folds != i) & (indices % n_folds != valid_fold)]
            splits.append({
                'train_idx': train_idx,
                'valid_idx': valid_idx,
                'test_idx': test_idx,
            })
        return splits

