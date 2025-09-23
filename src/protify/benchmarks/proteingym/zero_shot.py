import os
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from tqdm.auto import tqdm
from base_models.get_base_models import get_base_model
from .data_loader import load_proteingym_dms
from .scoring_utils import (
    label_row,
    get_optimal_window,
    get_sequence_slices,
    _parse_mutant_string,
    _apply_mutations_to_sequence,
    _position_log_probs,
    get_sequence_log_probability,
    calculate_pll,
)
from .scoring_utils import MODEL_CONTEXT_LENGTH

def zero_shot_scores_for_assay(
    df: pd.DataFrame,
    model_name: str,
    device: Optional[str] = None,
    progress: bool = True,
    tqdm_position: int = 1,
    scoring_method: str = "masked_marginal",
    scoring_window: str = "optimal" # "optimal" or "sliding"
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is empty")
    
    model, tokenizer = get_base_model(model_name, masked_lm=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()
    
    # Get sliced sequences
    target_seq = df['target_seq'].iloc[0]
    model_context_len = MODEL_CONTEXT_LENGTH.get(model_name, 1024)  # Default to 1024 if model not found
    if model_context_len is None:
        model_context_len = len(target_seq)
    sliced_df = get_sequence_slices(
        df, 
        target_seq=target_seq, 
        model_context_len=model_context_len, 
        start_idx=1, 
        scoring_window=scoring_window, 
        indel_mode=False
    )
    
    # Group sliced_df mutant to process each variant
    grouped = sliced_df.groupby('mutant')
    scores: List[float] = []
    
    iterator = df.iterrows()
    if progress:
        iterator = tqdm(
            iterator,
            total=len(df),
            desc="Assay variants",
            unit="variant",
            position=tqdm_position,
            leave=False,
        )
    
    log_prob_cache: Dict[Tuple[str, int, str], torch.Tensor] = {}
    pll_cache: Dict[str, Tuple[float, float]] = {}
    
    for _, row in iterator:
        mutant = row['mutant']
        mutated_seq = row['mutated_seq']
        muts = _parse_mutant_string(mutant)
        # Assert positions align with lengths
        for wt, pos, mt in muts:
            assert 0 <= pos < len(target_seq), (
                f"Mutation pos {pos} out of range for target_seq length {len(target_seq)}"
            )
        
        # Get all sliced seqs for this mutant
        mutant_slices = grouped.get_group(mutant)
        
        total_score = 0.0
        
        if scoring_method == "masked_marginal":
            mt_slices = mutant_slices[mutant_slices['mutated_seq'] == mutated_seq]
            
            for wt, pos, mt in muts:
                # Find the slice that contains this mutation position
                for _, slice_row in mt_slices.iterrows():
                    window_start = slice_row['window_start']
                    window_end = slice_row['window_end']
                    
                    if window_start <= pos < window_end:
                        window_seq = slice_row['sliced_mutated_seq']
                        pos_rel = pos - window_start
                        assert window_seq[pos_rel] == mutated_seq[pos], (
                            f"masked_marginal: residue mismatch at pos {pos} (rel {pos_rel})"
                        )
                        key_seq = window_seq[:pos_rel] + '<mask>' + window_seq[pos_rel+1:]
                        cache_key = (key_seq, pos_rel, scoring_method)
                        if cache_key not in log_prob_cache:
                            log_prob_cache[cache_key] = _position_log_probs(
                                model, tokenizer, scoring_method, window_seq, pos_rel, device
                            ).cpu()
                        lps = log_prob_cache[cache_key]
                        
                        # Build token_probs for label_row at this position
                        vocab_size = lps.shape[-1]
                        token_probs = torch.full((1, len(window_seq), vocab_size), fill_value=-1e9, dtype=lps.dtype)
                        token_probs[0, pos_rel, :] = lps
                        
                        # Compute score at the position relative to window
                        score = label_row(wt, pos_rel, mt, window_seq, token_probs, tokenizer)
                        total_score += float(score)
                        break
                        
        elif scoring_method == "mutant_marginal":
            # Feed full mutant sequence (no masking), compare scores between mutant and wildtype at mutated positions
            mt_slices = mutant_slices[mutant_slices['mutated_seq'] == mutated_seq]
            
            for wt, pos, mt in muts:
                # Find the slice that contains this mutation position
                for _, slice_row in mt_slices.iterrows():
                    window_start = slice_row['window_start']
                    window_end = slice_row['window_end']
                    
                    # Check if this mutation falls within this window
                    if window_start <= pos < window_end:
                        window_seq = slice_row['sliced_mutated_seq']
                        pos_rel = pos - window_start
                        assert window_seq[pos_rel] == mutated_seq[pos], (
                            f"mutant_marginal: residue mismatch at pos {pos} (rel {pos_rel})"
                        )
                        
                        cache_key = (window_seq, pos_rel, scoring_method)
                        if cache_key not in log_prob_cache:
                            log_prob_cache[cache_key] = _position_log_probs(
                                model, tokenizer, scoring_method, window_seq, pos_rel, device
                            ).cpu()
                        lps = log_prob_cache[cache_key]
                        
                        vocab_size = lps.shape[-1]
                        token_probs = torch.full((1, len(window_seq), vocab_size), fill_value=-1e9, dtype=lps.dtype)
                        token_probs[0, pos_rel, :] = lps
                        
                        # For mutant marginal: score wt vs mt when we have mt at pos_rel
                        score = label_row(mt, pos_rel, wt, window_seq, token_probs, tokenizer)
                        total_score -= float(score)  # Negate because we want mt - wt
                        break
                        
        elif scoring_method == "wildtype_marginal":
            # Feed full wildtype sequence (no masking), compare scores at mutated positions
            wt_slices = mutant_slices[mutant_slices['mutated_seq'] == target_seq]
            
            for wt, pos, mt in muts:
                # Find the slice that contains this mutation position
                for _, slice_row in wt_slices.iterrows():
                    window_start = slice_row['window_start']
                    window_end = slice_row['window_end']
                    
                    # Check if this mutation falls within this window
                    if window_start <= pos < window_end:
                        window_seq = slice_row['sliced_mutated_seq']
                        pos_rel = pos - window_start
                        assert window_seq[pos_rel] == target_seq[pos], (
                            f"wildtype_marginal: residue mismatch at pos {pos} (rel {pos_rel})"
                        )
                        
                        cache_key = (window_seq, pos_rel, scoring_method)
                        if cache_key not in log_prob_cache:
                            log_prob_cache[cache_key] = _position_log_probs(
                                model, tokenizer, scoring_method, window_seq, pos_rel, device
                            ).cpu()
                        lps = log_prob_cache[cache_key]
                        
                        vocab_size = lps.shape[-1]
                        token_probs = torch.full((1, len(window_seq), vocab_size), fill_value=-1e9, dtype=lps.dtype)
                        token_probs[0, pos_rel, :] = lps
                        
                        score = label_row(wt, pos_rel, mt, window_seq, token_probs, tokenizer)
                        total_score += float(score)
                        break
                        
        elif scoring_method == "pll":
            wt_slices = mutant_slices[mutant_slices['mutated_seq'] == target_seq]
            
            # Calculate PLL for all slices and average
            slice_scores = []
            for _, slice_row in wt_slices.iterrows():
                window_seq = slice_row['sliced_mutated_seq']
                ws, we = slice_row['window_start'], slice_row['window_end']
                assert window_seq == target_seq[ws:we], "PLL: slice content mismatch with window bounds"
                if window_seq not in pll_cache:
                    pll_cache[window_seq] = calculate_pll(window_seq, tokenizer, model, device)
                _, pll = pll_cache[window_seq]
                slice_scores.append(pll)
            total_score = sum(slice_scores) / len(slice_scores) if slice_scores else 0.0
            
        else:  # scoring_method == "global_log_prob"
            mt_slices = mutant_slices[mutant_slices['mutated_seq'] == mutated_seq]
            
            # Calculate log prob for all slices and sum
            for _, slice_row in mt_slices.iterrows():
                window_seq = slice_row['sliced_mutated_seq']
                ws, we = slice_row['window_start'], slice_row['window_end']
                assert window_seq == mutated_seq[ws:we], "global_log_prob: slice content mismatch with window bounds"
                seq_log_prob = get_sequence_log_probability(window_seq, tokenizer, model, device)
                total_score += seq_log_prob
        
        scores.append(total_score)
    
    out = df.copy()
    out['delta_log_prob'] = scores
    return out


def zero_shot_scores_for_indels(
    df: pd.DataFrame,
    model_name: str,
    device: Optional[str] = None,
    progress: bool = True,
    tqdm_position: int = 1,
    scoring_window: str = "sliding"
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is empty")

    model, tokenizer = get_base_model(model_name, masked_lm=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()

    target_seq = df['target_seq'].iloc[0]
    model_context_len = MODEL_CONTEXT_LENGTH.get(model_name, 1024)
    if model_context_len is None:
        model_context_len = len(target_seq)

    sliced_df = get_sequence_slices(
        df,
        target_seq=target_seq,
        model_context_len=model_context_len,
        start_idx=1,
        scoring_window='sliding',
        indel_mode=True
    )

    # Group by mutated sequence
    grouped = sliced_df.groupby('mutated_seq')

    scores: List[float] = []
    pll_cache: Dict[str, Tuple[float, float]] = {}

    iterator = df.iterrows()
    if progress:
        iterator = tqdm(
            iterator,
            total=len(df),
            desc="Assay variants (indels)",
            unit="variant",
            position=tqdm_position,
            leave=False,
        )

    for _, row in iterator:
        mutated_seq = row['mutated_seq']
        if mutated_seq not in grouped.groups:
            scores.append(0.0)
            continue
        mt_slices = grouped.get_group(mutated_seq)

        slice_scores = []
        for _, slice_row in mt_slices.iterrows():
            window_seq = slice_row['sliced_mutated_seq']
            if window_seq not in pll_cache:
                pll_cache[window_seq] = calculate_pll(window_seq, tokenizer, model, device)
            _, pll = pll_cache[window_seq]
            slice_scores.append(pll)
        total_score = sum(slice_scores) / len(slice_scores) if slice_scores else 0.0
        scores.append(total_score)

    out = df.copy()
    out['delta_log_prob'] = scores
    return out


def run_zero_shot(
    dms_ids: List[str],
    model_name: str,
    mode: str = "benchmark",
    repo_id: str = "GleghornLab/ProteinGym_DMS",
    results_dir: str = os.path.join('src', 'protify', 'results'),
    device: Optional[str] = None,
    show_progress: bool = True,
    scoring_method: str = "masked_marginal",
    scoring_window: str = "optimal",
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    assay_iterator = dms_ids
    if show_progress:
        assay_iterator = tqdm(dms_ids, desc="All assays", unit="assay", position=0)
    for dms_id in assay_iterator:
        df = load_proteingym_dms(dms_id, mode=mode, repo_id=repo_id)
        if df is None or len(df) == 0:
            continue
        if show_progress and hasattr(assay_iterator, 'set_description_str'):
            assay_iterator.set_description_str(f"Assay {dms_id}")
        if mode == 'indels':
            # Prefer sliding windows if any sequence exceeds context
            target_seq = df['target_seq'].iloc[0]
            model_context_len = MODEL_CONTEXT_LENGTH.get(model_name, 1024)
            max_len = max([len(target_seq)] + [len(s) for s in df['mutated_seq'].tolist()])
            results_df = zero_shot_scores_for_indels(
                df,
                model_name,
                device=device,
                progress=show_progress,
                tqdm_position=1,
                scoring_window='sliding'
            )
            suffix = 'pll'  # for file naming consistency
        else:
            results_df = zero_shot_scores_for_assay(
                df,
                model_name,
                device=device,
                progress=show_progress,
                tqdm_position=1,
                scoring_method=scoring_method,
                scoring_window=scoring_window
            )
            suffix = scoring_method
        # Aggregate per-assay predictions across models in a single CSV per DMS
        per_dms_path = os.path.join(results_dir, f"{dms_id}__zs_{suffix}.csv")

        # Prepare results for saving: rename score column to current model name to match config.json
        results_to_save = results_df.copy()
        if 'delta_log_prob' in results_to_save.columns:
            results_to_save = results_to_save.rename(columns={'delta_log_prob': model_name})
        # Only keep one row of 'target_seq' (present in first row), blank elsewhere
        first_target_seq = None
        if 'target_seq' in results_to_save.columns and len(results_to_save) > 0:
            first_target_seq = str(results_to_save['target_seq'].iloc[0])
            # Ensure exactly first row has the sequence; others blank
            results_to_save['target_seq'] = ''
            results_to_save.iloc[0, results_to_save.columns.get_loc('target_seq')] = first_target_seq

        # If an aggregated file exists for this DMS, append/merge the new model column
        if os.path.exists(per_dms_path):
            try:
                existing = pd.read_csv(per_dms_path)
                # Merge on 'mutant' if present; otherwise, align by row order
                if 'mutant' in existing.columns and 'mutant' in results_to_save.columns:
                    merged = existing.merge(
                        results_to_save[['mutant', model_name]],
                        on='mutant',
                        how='outer',
                    )
                # Ensure target_seq formatting: keep only in first row
                if 'target_seq' not in merged.columns:
                    # Create the column if missing
                    insert_at = merged.columns.get_loc('mutant') if 'mutant' in merged.columns else 0
                    merged.insert(insert_at, 'target_seq', '')
                if len(merged) > 0 and first_target_seq is not None:
                    merged.iloc[0, merged.columns.get_loc('target_seq')] = first_target_seq
                if len(merged) > 1:
                    merged.iloc[1:, merged.columns.get_loc('target_seq')] = ''
                merged.to_csv(per_dms_path, index=False)
            except Exception:
                # If anything goes wrong while merging, fall back to writing the current results
                results_to_save.to_csv(per_dms_path, index=False)
        else:
            # First model for this DMS: write a new aggregated file
            results_to_save.to_csv(per_dms_path, index=False)
        tqdm.write(f"[Assay {dms_id}] saved/updated: {per_dms_path}")
    return None