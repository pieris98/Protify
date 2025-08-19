import os
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional, Tuple
from tqdm.auto import tqdm
from base_models.get_base_models import get_base_model
from .data_loader import load_proteingym_dms
from .scoring_utils import (
    label_row,
    get_optimal_window,
    _parse_mutant_string,
    _masked_position_log_probs,
    get_sequence_log_probability,
)



def zero_shot_scores_for_assay(
    df: pd.DataFrame,
    model_name: str,
    device: Optional[str] = None,
    progress: bool = True,
    tqdm_position: int = 1,
    scoring_method: str = "masked",
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is empty")
    model, tokenizer = get_base_model(model_name)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    # Determine usable model window (exclude specials)
    model_config = getattr(model, 'config', None)
    if model_config is None and hasattr(model, 'esm') and hasattr(model.esm, 'config'):
        model_config = model.esm.config
    max_pos = int(getattr(model_config, 'max_position_embeddings', 1024))
    model_window = max(1, max_pos - 2)

    scores: List[float] = []
    row_iterator = df.iterrows()
    if progress:
        row_iterator = tqdm(
            row_iterator,
            total=len(df),
            desc="Assay variants",
            unit="variant",
            position=tqdm_position,
            leave=False,
        )
    if scoring_method == "unmasked":
        for _, row in row_iterator:
            wt_seq: str = row['target_seq']
            mut_seq: str = row['mutated_seq']
            if mut_seq is None or not isinstance(mut_seq, str) or len(mut_seq) == 0:
                scores.append(float('nan'))
                continue
            try:
                mut_lp = get_sequence_log_probability(mut_seq, tokenizer, model)
                wt_lp = get_sequence_log_probability(wt_seq, tokenizer, model)
                scores.append(float(mut_lp - wt_lp))
            except Exception:
                scores.append(float('nan'))
    else:
        # Determine usable model window (exclude specials)
        model_config = getattr(model, 'config', None)
        if model_config is None and hasattr(model, 'esm') and hasattr(model.esm, 'config'):
            model_config = model.esm.config
        max_pos = int(getattr(model_config, 'max_position_embeddings', 1024))
        model_window = max(1, max_pos - 2)
        # Cache masked-position log-probs by (window_seq, idx_rel_0)
        log_prob_cache: Dict[Tuple[str, int], torch.Tensor] = {}

        for _, row in row_iterator:
            wt_seq: str = row['target_seq']
            muts = _parse_mutant_string(row['mutant'])
            if len(muts) == 0:
                scores.append(float('nan'))
                continue
            total = 0.0
            for wt, pos, mt in muts:
                # Sanity checks
                if not (0 <= pos < len(wt_seq)):
                    continue
                if wt_seq[pos] != wt:
                    continue

                # Choose optimal window around the mutation site
                start, end = get_optimal_window(
                    mutation_position_relative=pos,
                    seq_len_wo_special=len(wt_seq),
                    model_window=model_window,
                )
                window_seq = wt_seq[start:end]
                pos_rel = pos - start

                cache_key = (window_seq, pos_rel)
                if cache_key not in log_prob_cache:
                    log_prob_cache[cache_key] = _masked_position_log_probs(model, tokenizer, window_seq, pos_rel, device).cpu()
                lps = log_prob_cache[cache_key]

                # Build token_probs for label_row at this position
                vocab_size = lps.shape[-1]
                token_probs = torch.full((1, len(window_seq), vocab_size), fill_value=-1e9, dtype=lps.dtype)
                token_probs[0, pos_rel, :] = lps

                # Compute score at the position relative to window
                score = label_row(wt, pos_rel, mt, window_seq, token_probs, tokenizer)
                total += float(score)
            scores.append(total)
    out = df.copy()
    out['delta_log_prob'] = scores
    return out


def run_zero_shot(
    dms_ids: List[str],
    model_name: str,
    mode: Optional[str] = None,
    repo_id: str = "nikraf/ProteinGym_DMS",
    results_dir: str = os.path.join('src', 'protify', 'results'),
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
    show_progress: bool = True,
    scoring_method: str = "masked",
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    assay_iterator = dms_ids
    if show_progress:
        assay_iterator = tqdm(dms_ids, desc="All assays", unit="assay", position=0)
    for dms_id in assay_iterator:
        df = load_proteingym_dms(dms_id, mode=mode, repo_id=repo_id, hf_token=hf_token)
        if df is None or len(df) == 0:
            continue
        if show_progress and hasattr(assay_iterator, 'set_description_str'):
            assay_iterator.set_description_str(f"Assay {dms_id}")
        results_df = zero_shot_scores_for_assay(
            df,
            model_name,
            device=device,
            progress=show_progress,
            tqdm_position=1,
            scoring_method=scoring_method,
        )
        # Aggregate per-assay predictions across models in a single CSV per DMS
        suffix = 'masked' if scoring_method == 'masked' else 'unmasked'
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
