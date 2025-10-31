import os
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any
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

        if scoring_method == "masked_marginal":
            # Create a Dict[(window_start, window_end, pos_tuple)] -> List[(row_idx, sorted_muts)]
            position_groups: Dict[Tuple[int, int, Tuple[int, ...]], List[Tuple[int, Tuple[Tuple[str, int, str], ...]]]] = {}
            
            for row_idx, row in enumerate(df.itertuples(index=False)):
                mutant = row.mutant
                muts = _parse_mutant_string(mutant)

                mutant_slices = mutant_groups.get(mutant)
                wt_slice = mutant_slices[mutant_slices['mutated_seq'] == target_seq]
                if len(wt_slice) == 0:
                    raise ValueError(f"No available slice for mutant {mutant} and method {scoring_method}")
                slice_row = wt_slice.iloc[0]

                window_start = int(slice_row['window_start'])
                window_end = int(slice_row['window_end'])

                # Sanity check
                min_pos = min(p for _, p, _ in muts)
                max_pos = max(p for _, p, _ in muts)
                if not (window_start <= min_pos and max_pos < window_end):
                    raise ValueError(
                        f"Window {window_start}-{window_end} does not contain all positions for variant {mutant}"
                    )

                # Sort mutations by absolute position so logits align with positions_list order
                sorted_muts = tuple(sorted(muts, key=lambda x: x[1])) # sorted_muts: (wt, pos, mt)
                pos_tuple = tuple(pos - window_start for _, pos, _ in sorted_muts) # pos_tuple: (rel_pos1, rel_pos2, ...)

                # Group by window and positions
                key = (window_start, window_end, pos_tuple)
                position_groups.setdefault(key, []).append((row_idx, sorted_muts)) 
            
            sequences: List[str] = [] # sequences to score
            positions_list: List[List[int]] = [] # List of positions to mask for each sequence
            variant_info: List[List[Tuple[int, List[Tuple[str, int, str]]]]] = [] # List of row indices and sorted mutations for each variant
            
            for (window_start, window_end, pos_tuple), variants in position_groups.items():
                window_seq = target_seq[window_start:window_end]
                sequences.append(window_seq)
                positions_list.append(list(pos_tuple))
                variant_info.append([(row_idx, list(sorted_muts)) for row_idx, sorted_muts in variants])
                
        elif scoring_method == "wildtype_marginal": # Group variants by window only - one forward pass per window
            # Create a Dict[(window_start, window_end)] -> List[(row_idx, sorted_muts, pos_rels)]
            window_groups: Dict[Tuple[int, int], List[Tuple[int, Tuple[Tuple[str, int, str], ...], Tuple[int, ...]]]] = {}
            
            for row_idx, row in enumerate(df.itertuples(index=False)):
                mutant = row.mutant
                muts = _parse_mutant_string(mutant)

                mutant_slices = mutant_groups.get(mutant)
                wt_slice = mutant_slices[mutant_slices['mutated_seq'] == target_seq]
                if len(wt_slice) == 0:
                    raise ValueError(f"No available slice for mutant {mutant} and method {scoring_method}")
                slice_row = wt_slice.iloc[0]

                window_start = int(slice_row['window_start'])
                window_end = int(slice_row['window_end'])

                # Sanity check
                min_pos = min(p for _, p, _ in muts)
                max_pos = max(p for _, p, _ in muts)
                if not (window_start <= min_pos and max_pos < window_end):
                    raise ValueError(
                        f"Window {window_start}-{window_end} does not contain all positions for variant {mutant}"
                    )

                # Sort mutations by absolute position so logits align with positions_list order
                sorted_muts = tuple(sorted(muts, key=lambda x: x[1]))
                pos_rels = tuple(pos - window_start for _, pos, _ in sorted_muts)

                # Group by window only
                key = (window_start, window_end)
                window_groups.setdefault(key, []).append((row_idx, sorted_muts, pos_rels))
            
            sequences: List[str] = [] # sequences to score
            positions_list: List[List[int]] = [] # List of positions to score for each sequence
            window_to_variants: List[List[Tuple[int, List[Tuple[str, int, str]], List[int]]]] = [] # List of row indices, sorted mutations, and relative positions for each variant in each window
            
            for (window_start, window_end), variants in window_groups.items():
                window_seq = target_seq[window_start:window_end]
                sequences.append(window_seq)
                
                # Collect all unique positions needed for this window
                all_positions = set()
                for _, _, pos_rels in variants:
                    all_positions.update(pos_rels)
                positions_list.append(sorted(all_positions))
                # Store all variants' info for this window
                window_to_variants.append([(row_idx, list(sorted_muts), list(pos_rels)) for row_idx, sorted_muts, pos_rels in variants])
                
        else:  # mutant_marginal - each variant needs to be processed independently
            sequences: List[str] = [] # sequences to score
            positions_list: List[List[int]] = [] # List of positions to score for each sequence
            variant_info: List[Tuple[int, List[Tuple[str, int, str]]]] = []  # List of row indices and muts for each variant

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
                mut_slice = mutant_slices[mutant_slices['mutated_seq'] == mutated_seq]
                if len(mut_slice) == 0:
                    raise ValueError(f"No available slice for mutant {mutant} and method {scoring_method}")
                slice_row = mut_slice.iloc[0]

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
        if scoring_method == "masked_marginal":
            total_variants = len(df)
            print(f"Computing scores for {len(sequences)} inputs, covering {total_variants} variants ...")
        elif scoring_method == "wildtype_marginal":
            total_variants = len(df)
            print(f"Computing scores for {len(sequences)} windows, covering {total_variants} variants ...")
        else:  # mutant_marginal
            print(f"Computing scores for {len(sequences)} variants ...")
        
        iterator = range(0, len(sequences), batch_size)
        if progress:
            iterator = tqdm(
                iterator,
                total=(len(sequences) + batch_size - 1) // batch_size,
                desc=f"Assay batches ({scoring_method})",
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
        
        if scoring_method == "masked_marginal":
            for variants_in_group, score in zip(variant_info, per_variant_log_probs): # score shape: [num_positions, vocab]
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
                    
        elif scoring_method == "wildtype_marginal":
            # For wildtype_marginal, per_variant_log_probs contains log probs for ALL positions in each window
            # We need to map each variant to its window and extract only its positions
            for window_idx, (window_log_probs, variants) in enumerate(zip(per_variant_log_probs, window_to_variants)):
                # window_log_probs shape: [num_all_positions_in_window, vocab]
                # Create a mapping from relative position to index in window_log_probs
                window_positions = positions_list[window_idx]
                pos_to_idx = {pos: idx for idx, pos in enumerate(window_positions)} # map rel_pos to idx in window_log_probs tensor
                
                for row_idx, muts, pos_rels in variants:
                    # Extract log probs only for this variant's positions
                    pos_indices = torch.tensor([pos_to_idx[pos] for pos in pos_rels], device=window_log_probs.device)
                    variant_log_probs = window_log_probs[pos_indices]  # [num_mutations, vocab]
                    
                    assert variant_log_probs.size(0) == len(muts), "Mismatch between mutations and gathered logits"
                    wt_ids, mt_ids = [], []
                    for wt, _pos, mt in muts:
                        wt_id = aa_to_id.get(wt)
                        mt_id = aa_to_id.get(mt)
                        if wt_id is None or mt_id is None or (unk_id is not None and (wt_id == unk_id or mt_id == unk_id)):
                            raise ValueError(f"WT or MT is not in vocab: {wt} or {mt}")
                        wt_ids.append(wt_id)
                        mt_ids.append(mt_id)

                    wt_tensor = torch.as_tensor(wt_ids, device=variant_log_probs.device)
                    mt_tensor = torch.as_tensor(mt_ids, device=variant_log_probs.device)
                    indices = torch.arange(len(wt_ids), device=variant_log_probs.device)
                    deltas = variant_log_probs[indices, mt_tensor] - variant_log_probs[indices, wt_tensor]
                    scores[row_idx] = deltas.sum().item()
                    
        else:  # mutant_marginal
            for (row_idx, muts), score in zip(variant_info, per_variant_log_probs): # score shape: [num_mutations, vocab]
                assert score.size(0) == len(muts), "Mismatch between mutations and gathered logits"
                muts = sorted(muts, key=lambda x: x[1])
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
        mutated_slices = sliced_df[sliced_df['mutated_seq'] != target_seq].copy()
        
        # Collect seqs - deduplicate by sequence for efficiency
        seqs_to_score = mutated_slices['sliced_mutated_seq'].drop_duplicates().tolist()
        
        # Batch compute PLL for all unique seqs
        print(f"Computing PLL for {len(seqs_to_score)} unique sequences...")
        
        pll_progress = None
        if progress:
            pll_progress = tqdm(
                total=len(seqs_to_score),  # Will be updated as position batches are processed
                desc="PLL computation",
                unit="seq",
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
            progress_bar=pll_progress
        )
        
        if pll_progress is not None:
            pll_progress.close()
        
        # Create a mapping from sequence to PLL score
        seq_to_pll = {seq: res[0] for seq, res in zip(seqs_to_score, pll_results)}
        
        # Map PLL scores back to mutated_slices
        mutated_slices['sequence_pll'] = mutated_slices['sliced_mutated_seq'].map(seq_to_pll)
        
        # One score per variant; do not average across windows
        scores_by_variant = (
            mutated_slices.groupby('mutant')['sequence_pll']
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
    pll_cache = {seq: result for seq, result in zip(seqs_to_score, pll_results)}
    
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
