import os
import json
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def _load_parquet_by_dms(repo_id: str, dms_id: str) -> Optional[pd.DataFrame]:
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
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        except Exception:
            continue
        try:
            df = pd.read_parquet(local_path)
            return df
        except Exception:
            continue
    return None


def _load_via_index_select(repo_id: str, dms_id: str) -> Optional[pd.DataFrame]:
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
    base = load_dataset(repo_id, split="train", streaming=False)
    try:
        subset = base.select(indices)
        df = subset.to_pandas().reset_index(drop=True)
        return df
    except Exception:
        return None


def load_proteingym_dms(dms_id: str, mode: Optional[str] = None, repo_id: str = "GleghornLab/ProteinGym_DMS") -> pd.DataFrame:
    """
    Loads a single ProteinGym DMS assay from Hugging Face.
    """
    # per-assay parquet shard
    df = _load_parquet_by_dms(repo_id=repo_id, dms_id=dms_id)
    if df is None:
        # use precomputed index to select
        df = _load_via_index_select(repo_id=repo_id, dms_id=dms_id)
    if df is None:
        # try streaming filter to avoid materializing full dataset
        try:
            hf_stream = load_dataset(repo_id, split="train", streaming=True)
            rows = [row for row in hf_stream if row.get("DMS_id", None) == dms_id]
            df = pd.DataFrame.from_records(rows)
        except Exception:
            df = None
    if df is None or len(df) == 0:
        raise ValueError(f"No data loaded for DMS_id={dms_id}")
    
    if mode == 'zero_shot': # single & multiple mutants. Standard zero-shot DMS ProteinGym
        df = df[df['is_indel'] == False]
        df = df[["DMS_id","mutated_seq","target_seq","DMS_score",
                 "DMS_score_bin","mutant"]] 
    elif mode == 'singles_zero_shot': # single mutants only. Standard zero-shot DMS ProteinGym
        df = df[df['num_mutations'] == 1]
        df = df[["DMS_id","mutated_seq","target_seq","DMS_score",
                 "DMS_score_bin","mutant"]]
    elif mode == 'indels_zero_shot': # indels only zero-shot. Standard zero-shot DMS Indels
        df = df[df['is_indel'] == True]
        df = df[["DMS_id","mutated_seq","target_seq","DMS_score",
                 "DMS_score_bin"]]
    elif mode == 'multiples_zero_shot': # multiple mutants only zero-shot
        df = df[df['num_mutations'] > 1]
        df = df[["DMS_id","mutated_seq","target_seq","DMS_score",
                 "DMS_score_bin","mutant"]]
    elif mode == 'supervised': # single mutants only. Standard supervised DMS ProteinGym
        df = df[df['is_indel'] == False]
        df = df[["DMS_id","mutated_seq","target_seq","DMS_score",
                 "DMS_score_bin","mutant","fold_random_5","fold_contiguous_5",
                 "fold_modulo_5"]]
    elif mode == 'supervised_multiples': # multiple mutants only
        df = df[df['is_indel'] == False]
        df = df[df['num_mutations'] > 1]
        df = df[["DMS_id","mutated_seq","target_seq","DMS_score",
                 "DMS_score_bin","mutant","fold_rand_multiples"]]
    elif mode == 'indels_supervised': # indels only supervised. Standard supervised DMS Indels
        df = df[df['is_indel'] == True]
        df = df[["DMS_id","mutated_seq","target_seq","DMS_score",
                 "DMS_score_bin","fold_random_5_indels"]]

    return df.reset_index(drop=True)