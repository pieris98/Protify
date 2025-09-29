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
    get_sequence_log_probability_batched,
    calculate_pll_batched,
    _precompute_aa_token_ids,
)
from .scoring_utils import MODEL_CONTEXT_LENGTH


def zero_shot_scores_for_assay(
    df: pd.DataFrame,
    model_name: str,
    device: str = None,
    progress: bool = True,
    tqdm_position: int = 1,
    scoring_method: str = "masked_marginal",
    scoring_window: str = "optimal", # "optimal" or "sliding"
    model = None,
    tokenizer = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is empty")
    
    # Get sliced sequences
    target_seq = df['target_seq'].iloc[0]
    model_context_len = MODEL_CONTEXT_LENGTH.get(model_name, 1024)
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
    
    # Group sliced_df by mutant to process each variant
    grouped = sliced_df.groupby('mutant')
    
    if scoring_method in ["masked_marginal", "mutant_marginal", "wildtype_marginal"]:
        # Precompute amino acid token IDs for lookup
        aa_to_id = _precompute_aa_token_ids(tokenizer)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        
        # Step 1: Collect all unique (sequence, position) pairs that need computation
        forwards = {}  # Maps (window_seq, pos_rel) -> list of (row_idx, mutation_info)
        variants = []  # List of (row_idx, mutant, mutated_seq, parsed_mutations)
        mutation_locs = {}  # Maps (row_idx, pos) -> (window_seq, pos_rel)
        
        for row_idx, row in df.iterrows():
            mutant = row['mutant']
            mutated_seq = row['mutated_seq']
            muts = _parse_mutant_string(mutant)
            
            # Assert positions align with lengths
            for wt, pos, mt in muts:
                assert 0 <= pos < len(target_seq), (
                    f"Mutation pos {pos} out of range for target_seq length {len(target_seq)}"
                )
            
            variants.append((row_idx, mutant, mutated_seq, muts))
            
            # Get all sliced seqs for this mutant
            mutant_slices = grouped.get_group(mutant)
            
            # Determine which slices to use based on scoring method
            if scoring_method in ["masked_marginal", "mutant_marginal"]:
                slices_to_use = mutant_slices[mutant_slices['mutated_seq'] == mutated_seq]
            else:  # wildtype_marginal
                slices_to_use = mutant_slices[mutant_slices['mutated_seq'] == target_seq]
            
            # For each mutation, find the window that contains it
            for wt, pos, mt in muts:
                for _, slice_row in slices_to_use.iterrows():
                    window_start = slice_row['window_start']
                    window_end = slice_row['window_end']
                    
                    if window_start <= pos < window_end:
                        window_seq = slice_row['sliced_mutated_seq']
                        pos_rel = pos - window_start
                        
                        # Verify sequence consistency
                        if scoring_method in ["masked_marginal", "mutant_marginal"]:
                            assert window_seq[pos_rel] == mutated_seq[pos], (
                                f"{scoring_method}: residue mismatch at pos {pos} (rel {pos_rel})"
                            )
                        else:  # wildtype_marginal
                            assert window_seq[pos_rel] == target_seq[pos], (
                                f"wildtype_marginal: residue mismatch at pos {pos} (rel {pos_rel})"
                            )
                        
                        key = (window_seq, pos_rel)
                        if key not in forwards:
                            forwards[key] = []
                        forwards[key].append((row_idx, wt, pos, mt))
                        # Store mapping for reuse when assigning scores
                        mutation_locs[(row_idx, pos)] = key
                        break
        
        # Step 2: Batch compute log probabilities for all unique (sequence, position) pairs
        unique_pairs = list(forwards.keys())
        sequences = [pair[0] for pair in unique_pairs]
        positions = [pair[1] for pair in unique_pairs]
        
        print(f"Computing {len(unique_pairs)} unique sequence-position pairs in batches...")
        
        all_log_probs = _position_log_probs(
            model, tokenizer, scoring_method, sequences, positions, 
            device, model_name, batch_size=batch_size
        )
        
        # Create a mapping from (window_seq, pos_rel) to log_probs
        log_prob_cache = {}
        for i, key in enumerate(unique_pairs):
            log_prob_cache[key] = all_log_probs[i]
        
        # Step 3: Assign scores to all variants
        scores = [0.0] * len(df)
        
        for row_idx, mutant, mutated_seq, muts in variants:
            total_score = 0.0
                        
            for wt, pos, mt in muts:
                key = mutation_locs.get((row_idx, pos))
                if key is None:
                    raise KeyError(f"Missing precomputed slice for row {row_idx}, position {pos}")
                lps = log_prob_cache[key]

                # Use precomputed token IDs
                wt_id = aa_to_id.get(wt)
                mt_id = aa_to_id.get(mt)

                if wt_id is None or mt_id is None or (unk_id is not None and (wt_id == unk_id or mt_id == unk_id)):
                    raise ValueError(f"WT or MT is not in vocab: {wt} or {mt}")

                score = (lps[mt_id] - lps[wt_id]).item()
                total_score += float(score)
            
            scores[row_idx] = total_score
        
    elif scoring_method == "pll":
        # Collect all unique sequences that need PLL computation
        unique_sequences = set()
        for _, row in df.iterrows():
            mutant = row['mutant']
            mutant_slices = grouped.get_group(mutant)
            wt_slices = mutant_slices[mutant_slices['mutated_seq'] == target_seq]
            
            for _, slice_row in wt_slices.iterrows():
                window_seq = slice_row['sliced_mutated_seq']
                unique_sequences.add(window_seq)
        
        # Batch compute PLL for all unique sequences
        print(f"Computing PLL for {len(unique_sequences)} unique sequences...")
        unique_seq_list = list(unique_sequences)
        pll_results = calculate_pll_batched(
            unique_seq_list,
            tokenizer,
            model,
            device,
            model_name,
            batch_size=batch_size
        )
        
        # Create cache
        pll_cache = {seq: result for seq, result in zip(unique_seq_list, pll_results)}
        
        # Assign scores
        scores = []
        iterator = df.iterrows()
        if progress:
            iterator = tqdm(
                iterator,
                total=len(df),
                desc="Assay variants (PLL)",
                unit="variant",
                position=tqdm_position,
                leave=False,
            )
        
        for _, row in iterator:
            mutant = row['mutant']
            mutant_slices = grouped.get_group(mutant)
            wt_slices = mutant_slices[mutant_slices['mutated_seq'] == target_seq]
            
            slice_scores = []
            for _, slice_row in wt_slices.iterrows():
                window_seq = slice_row['sliced_mutated_seq']
                _, pll = pll_cache[window_seq]
                slice_scores.append(pll)
            
            total_score = sum(slice_scores) / len(slice_scores) if slice_scores else 0.0
            scores.append(total_score)
    
    else:  # scoring_method == "global_log_prob"
        # Collect all unique sequences that need log prob computation
        unique_sequences = set()
        sequence_to_rows = {}  # Map sequences to rows that use them
        
        for row_idx, row in df.iterrows():
            mutant = row['mutant']
            mutated_seq = row['mutated_seq']
            mutant_slices = grouped.get_group(mutant)
            mt_slices = mutant_slices[mutant_slices['mutated_seq'] == mutated_seq]
            
            for _, slice_row in mt_slices.iterrows():
                window_seq = slice_row['sliced_mutated_seq']
                unique_sequences.add(window_seq)
                if window_seq not in sequence_to_rows:
                    sequence_to_rows[window_seq] = []
                sequence_to_rows[window_seq].append(row_idx)
        
        # Batch compute log probabilities for all unique sequences
        print(f"Computing global log prob for {len(unique_sequences)} unique sequences...")
        unique_seq_list = list(unique_sequences)
        log_prob_results = get_sequence_log_probability_batched(
            unique_seq_list,
            tokenizer,
            model,
            device,
            model_name,
            batch_size=batch_size
        )
        
        # Create cache
        log_prob_cache = {seq: result for seq, result in zip(unique_seq_list, log_prob_results)}
        
        # Assign scores
        scores = [0.0] * len(df)
        iterator = df.iterrows()
        if progress:
            iterator = tqdm(
                iterator,
                total=len(df),
                desc="Assay variants (global_log_prob)",
                unit="variant",
                position=tqdm_position,
                leave=False,
            )
        
        for row_idx, row in iterator:
            mutant = row['mutant']
            mutated_seq = row['mutated_seq']
            mutant_slices = grouped.get_group(mutant)
            mt_slices = mutant_slices[mutant_slices['mutated_seq'] == mutated_seq]
            
            total_score = 0.0
            for _, slice_row in mt_slices.iterrows():
                window_seq = slice_row['sliced_mutated_seq']
                seq_log_prob = log_prob_cache[window_seq]
                total_score += seq_log_prob
            
            scores[row_idx] = total_score
    
    out = df.copy()
    out['delta_log_prob'] = scores
    return out



def zero_shot_scores_for_indels(
    df: pd.DataFrame,
    model_name: str,
    device: Optional[str] = None,
    progress: bool = True,
    tqdm_position: int = 1,
    scoring_window: str = "sliding",
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is empty")

    if model is None or tokenizer is None:
        model, tokenizer = get_base_model(model_name, masked_lm=True)
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(device).eval()
    else:
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

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
                pll_cache[window_seq] = calculate_pll_batched(window_seq, tokenizer, model, device, model_name, batch_size=batch_size)
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
    batch_size: int = 32,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    assay_iterator = dms_ids
    if show_progress:
        assay_iterator = tqdm(dms_ids, desc="All assays", unit="assay", position=0)
    # Load model once per model_name to avoid repeated initialization
    model, tokenizer = get_base_model(model_name, masked_lm=True)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(torch_device).eval()

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
                device=torch_device,
                progress=show_progress,
                tqdm_position=1,
                scoring_window='sliding',
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
            )
            suffix = 'pll'  # for file naming consistency
        else:
            results_df = zero_shot_scores_for_assay(
                df,
                model_name,
                device=torch_device,
                progress=show_progress,
                tqdm_position=1,
                scoring_method=scoring_method,
                scoring_window=scoring_window,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
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
                # Merge priority: mutant -> mutated_seq -> row order
                if 'mutant' in existing.columns:
                    merged = existing.merge(
                        results_to_save[['mutant', model_name]],
                        on='mutant',
                        how='outer',
                    )
                elif 'mutated_seq' in existing.columns:
                    merged = existing.merge(
                        results_to_save[['mutated_seq', model_name]],
                        on='mutated_seq',
                        how='outer',
                    )
                ## Ensure target_seq formatting: keep only in first row
                #if 'target_seq' not in merged.columns:
                #    insert_at = merged.columns.get_loc('mutant') if 'mutant' in merged.columns else 0
                #    merged.insert(insert_at, 'target_seq', '')
                #if len(merged) > 0 and first_target_seq is not None:
                #    merged.iloc[0, merged.columns.get_loc('target_seq')] = first_target_seq
                #if len(merged) > 1:
                #    merged.iloc[1:, merged.columns.get_loc('target_seq')] = ''
                merged.to_csv(per_dms_path, index=False)
            except Exception as e:
                print(f"Error merging results for {dms_id}: {e}")
                # If anything goes wrong while merging, fall back to writing the current results
                results_to_save.to_csv(per_dms_path, index=False)
        else:
            # First model for this DMS: write a new aggregated file
            results_to_save.to_csv(per_dms_path, index=False)
        tqdm.write(f"[Assay {dms_id}] saved/updated: {per_dms_path}")
    return None