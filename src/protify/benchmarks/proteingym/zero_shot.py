import os
import pandas as pd
import torch
from typing import List, Optional, Tuple, Any
from tqdm.auto import tqdm
from base_models.get_base_models import get_base_model
from .data_loader import load_proteingym_dms
from .scoring_utils import (
    get_sequence_slices,
    _parse_mutant_string,
    _position_log_probs,
    get_sequence_log_probability_batched,
    calculate_pll_batched,
    _aa_to_token_ids,
)
from .scoring_utils import MODEL_CONTEXT_LENGTH
from collections import defaultdict


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
    mutant_groups = {mutant: group for mutant, group in grouped}
    
    if scoring_method in ["masked_marginal", "mutant_marginal", "wildtype_marginal"]:
        # Precompute amino acid token IDs for lookup
        aa_to_id = _aa_to_token_ids(tokenizer)
        unk_id = getattr(tokenizer, "unk_token_id", None)

        if scoring_method in ["masked_marginal", "wildtype_marginal"]: # Group variants by mutation positions            
            position_groups = defaultdict(list)  # key: (window_start, window_end, tuple(positions)), value: list of (row_idx, muts)
            
            for row_idx, row in enumerate(df.itertuples(index=False)):
                mutant = row.mutant
                mutated_seq = row.mutated_seq
                muts = _parse_mutant_string(mutant)

                # Sanity check
                for wt, pos, mt in muts:
                    assert 0 <= pos < len(target_seq), (
                        f"Mutation pos {pos} out of range for target_seq length {len(target_seq)}"
                    )

                mutant_slices = mutant_groups.get(mutant)
                if mutant_slices is None:
                    raise ValueError(f"No slices available for mutant {mutant}")
                slices_to_use = mutant_slices[mutant_slices['mutated_seq'] == target_seq]

                # Use first available slice (for optimal, just first row)
                if len(slices_to_use) == 0:
                    raise ValueError(f"No available slice for mutant {mutant} and method {scoring_method}")
                slice_row = slices_to_use.iloc[0]

                window_start = int(slice_row['window_start'])
                window_end = int(slice_row['window_end'])
                window_seq = slice_row['sliced_mutated_seq']

                # Ensure window contains ALL mutated positions (sanity check)
                min_pos = min(p for _, p, _ in muts)
                max_pos = max(p for _, p, _ in muts)
                if not (window_start <= min_pos and max_pos < window_end):
                    raise ValueError(
                        f"Window {window_start}-{window_end} does not contain all positions for variant {mutant}"
                    )

                # Build list of relative positions for each mutation based on window start
                pos_rels: List[int] = []
                for wt, pos, mt in muts:
                    rel = pos - window_start
                    assert window_seq[rel] == target_seq[pos], (
                        f"{scoring_method}: residue mismatch at abs {pos} (rel {rel})"
                    )
                    pos_rels.append(rel)

                # Group by window and positions to prevent duplicate computations
                key = (window_start, window_end, tuple(pos_rels), window_seq)
                position_groups[key].append((row_idx, muts))
            
            # Process each unique position group
            sequences: List[str] = [] # sequences to score
            positions_list: List[List[int]] = [] # List of positions to score/mask for each sequence
            variant_info: List[List[Tuple[int, List[Tuple[str, int, str]]]]] = []  # List of row indices and mutations for each variant
            
            for (window_start, window_end, pos_tuple, window_seq), variants in position_groups.items():
                sequences.append(window_seq)
                positions_list.append(list(pos_tuple))
                variant_info.append(variants)
                
        else:  # mutant_marginal - each variant needs to be processed independently
            sequences: List[str] = [] # sequences to score
            positions_list: List[List[int]] = [] # List of positions to score for each sequence
            variant_info: List[Tuple[int, List[Tuple[str, int, str]]]] = []  # List of row indices and mutations for each variant

            for row_idx, row in enumerate(df.itertuples(index=False)):
                mutant = row.mutant
                mutated_seq = row.mutated_seq
                muts = _parse_mutant_string(mutant)

                # Validate positions
                for wt, pos, mt in muts:
                    assert 0 <= pos < len(target_seq), (
                        f"Mutation pos {pos} out of range for target_seq length {len(target_seq)}"
                    )

                mutant_slices = mutant_groups.get(mutant)
                if mutant_slices is None:
                    raise ValueError(f"No slices available for mutant {mutant}")
                slices_to_use = mutant_slices[mutant_slices['mutated_seq'] == mutated_seq]

                # Use the first available slice (for optimal, there should be exactly one)
                if len(slices_to_use) == 0:
                    raise ValueError(f"No available slice for mutant {mutant} and method {scoring_method}")
                slice_row = slices_to_use.iloc[0]

                window_start = int(slice_row['window_start'])
                window_end = int(slice_row['window_end'])
                window_seq = slice_row['sliced_mutated_seq']

                # Ensure the window contains ALL mutated positions
                min_pos = min(p for _, p, _ in muts)
                max_pos = max(p for _, p, _ in muts)
                if not (window_start <= min_pos and max_pos < window_end):
                    raise ValueError(
                        f"Window {window_start}-{window_end} does not contain all positions for variant {mutant}"
                    )

                # Build a list of relative positions for each mutation based on the window start
                pos_rels: List[int] = []
                for wt, pos, mt in muts:
                    rel = pos - window_start
                    assert window_seq[rel] == mutated_seq[pos], (
                        f"mutant_marginal: residue mismatch at abs {pos} (rel {rel})"
                    )
                    pos_rels.append(rel)

                sequences.append(window_seq)
                positions_list.append(pos_rels)
                variant_info.append((row_idx, muts))

        # Compute logits for ALL positions per sequence
        print(f"Computing scores for {len(sequences)} variants ...")
        
        iterator = range(0, len(sequences), batch_size)
        if progress:
            iterator = tqdm(
                iterator,
                total=(len(sequences) + batch_size - 1) // batch_size,
                desc=f"Assay batches ({scoring_method} multi-pos)",
                unit="batch",
                position=tqdm_position,
                leave=False,
            )

        per_variant_log_probs = _position_log_probs(
            model,
            tokenizer,
            scoring_method,
            sequences,
            positions_list,
            device,
            model_name,
            batch_size=batch_size,
            progress_bar=iterator if progress else None,
        )

        # Assign scores per variant
        scores = [0.0] * len(df)
        
        if scoring_method in ["masked_marginal", "wildtype_marginal"]:
            for variants_in_group, score in zip(variant_info, per_variant_log_probs): # score shape: [num_mutations, vocab]
                for row_idx, muts in variants_in_group:
                    assert score.size(0) == len(muts), "Mismatch between mutations and gathered logits"
                    wt_ids, mt_ids = [], []
                    for wt, _pos, mt in muts:
                        wt_id = aa_to_id.get(wt)
                        mt_id = aa_to_id.get(mt)
                        if wt_id is None or mt_id is None or (unk_id is not None and (wt_id == unk_id or mt_id == unk_id)):
                            raise ValueError(f"WT or MT is not in vocab: {wt} or {mt}")
                        wt_ids.append(wt_id)
                        mt_ids.append(mt_id)

                    wt_tensor = torch.as_tensor(wt_ids, device=score.device)
                    mt_tensor = torch.as_tensor(mt_ids, device=score.device)
                    indices = torch.arange(len(wt_ids), device=score.device)
                    deltas = score[indices, mt_tensor] - score[indices, wt_tensor]
                    scores[row_idx] = deltas.sum().item()
        else:
            for (row_idx, muts), score in zip(variant_info, per_variant_log_probs): # score shape: [num_mutations, vocab]
                assert score.size(0) == len(muts), "Mismatch between mutations and gathered logits"
                wt_ids, mt_ids = [], []
                for wt, _pos, mt in muts:
                    wt_id = aa_to_id.get(wt)
                    mt_id = aa_to_id.get(mt)
                    if wt_id is None or mt_id is None or (unk_id is not None and (wt_id == unk_id or mt_id == unk_id)):
                        raise ValueError(f"WT or MT is not in vocab: {wt} or {mt}")
                    wt_ids.append(wt_id)
                    mt_ids.append(mt_id)

                wt_tensor = torch.as_tensor(wt_ids, device=score.device)
                mt_tensor = torch.as_tensor(mt_ids, device=score.device)
                indices = torch.arange(len(wt_ids), device=score.device)
                deltas = score[indices, mt_tensor] - score[indices, wt_tensor]
                scores[row_idx] = deltas.sum().item()
        
    elif scoring_method == "pll":
        wt_slices = sliced_df[sliced_df['mutated_seq'] == target_seq].copy()
        
        # Collect seqs
        seqs_to_score = wt_slices['sliced_mutated_seq']
        
        # Batch compute PLL for all seqs
        print(f"Computing PLL for {len(seqs_to_score)} unique sequences...")
        
        iterator = range(0, len(seqs_to_score), batch_size)
        if progress:
            iterator = tqdm(
                iterator,
                total=(len(seqs_to_score) + batch_size - 1) // batch_size,
                desc="PLL batches",
                unit="batch",
                position=tqdm_position,
                leave=False,
            )
        
        pll_results = calculate_pll_batched(
            seqs_to_score,
            tokenizer,
            model,
            device,
            model_name,
            batch_size=batch_size,
            progress_bar=iterator if progress else None
        )
        
        # Append per-sequence PLL (total) directly, aligned by row order
        wt_slices = wt_slices.copy()
        wt_slices['sequence_pll'] = [res[0] for res in pll_results]
        
        # One score per variant; do not average across windows
        scores_by_variant = (
            wt_slices.groupby('mutant')['sequence_pll']
              .first()
              .to_dict()
        )
        
    
    else:  # scoring_method == "global_log_prob"
        # Score only mutated sequence slices
        mutated_slices = sliced_df[sliced_df['mutated_seq'] != target_seq].copy()
        seqs_to_score = mutated_slices['sliced_mutated_seq'].tolist()
        
        # Batch compute log probabilities for all seqs
        print(f"Computing global log prob for {len(seqs_to_score)} unique sequences...")
        
        iterator = range(0, len(seqs_to_score), batch_size)
        if progress:
            iterator = tqdm(
                iterator,
                total=(len(seqs_to_score) + batch_size - 1) // batch_size,
                desc="Global log prob batches",
                unit="batch",
                position=tqdm_position,
                leave=False,
            )
        
        log_prob_results = get_sequence_log_probability_batched(
            seqs_to_score,
            tokenizer,
            model,
            device,
            model_name,
            batch_size=batch_size,
            progress_bar=iterator if progress else None
        )
        
        # Append per-sequence log prob directly, aligned by row order
        mutated_slices['sequence_log_prob'] = log_prob_results
        
        # One score per variant; do not aggregate across windows
        scores_by_variant = (
            mutated_slices.groupby('mutant')['sequence_log_prob']
              .first()
              .to_dict()
        )
        
    
    out = df.copy()
    if scoring_method in ["masked_marginal", "mutant_marginal", "wildtype_marginal"]:
        out['delta_log_prob'] = scores
    else:
        out['delta_log_prob'] = out['mutant'].map(scores_by_variant)
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

    seqs_to_score = sliced_df['sliced_mutated_seq'].to_list()
    
    # Batch compute PLL for all unique sequences
    print(f"Computing PLL for {len(seqs_to_score)} unique sequences (indels)...")
            
    # Add progress bar for PLL computation
    iterator = range(0, len(seqs_to_score), batch_size)
    if progress:
        iterator = tqdm(
            iterator,
            total=(len(seqs_to_score) + batch_size - 1) // batch_size,
            desc="PLL batches (indels)",
            unit="batch",
            position=tqdm_position,
            leave=False,
        )
    
    pll_results = calculate_pll_batched(
        seqs_to_score,
        tokenizer,
        model,
        device,
        model_name,
        batch_size=batch_size,
        progress_bar=iterator if progress else None
    )
    
    # Grab normalized PLL for indels
    pll_cache = {seq: result[1] for seq, result in zip(seqs_to_score, pll_results)}
    
    # Add a mapped column of per-window scores, then average by mutated_seq
    sliced_df['window_score'] = sliced_df['sliced_mutated_seq'].map(pll_cache)
    scores_by_variant = (
        sliced_df.groupby('mutated_seq')['window_score']
          .mean()
          .to_dict()
    )
    

    out = df.copy()
    out['delta_log_prob'] = out['mutated_seq'].map(scores_by_variant)
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
