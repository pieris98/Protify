import os
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional, Tuple
from scipy.stats import spearmanr, pearsonr
from transformers import AutoModelForMaskedLM, AutoTokenizer
from ...base_models.get_base_models import get_base_model
from .data_loader import load_proteingym_dms
from .scoring_utils import (
    label_row,
    get_optimal_window,
    _parse_mutant_string,
    _masked_position_log_probs,
)



def zero_shot_masked_scores_for_df(df: pd.DataFrame, model_name: str, device: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is empty")
    model_path = get_base_model(model_name)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Determine usable model window (exclude specials)
    max_pos = int(getattr(model.config, 'max_position_embeddings', 1024))
    model_window = max(1, max_pos - 2)

    # Cache masked-position log-probs by (window_seq, idx_rel_0)
    log_prob_cache: Dict[Tuple[str, int], torch.Tensor] = {}

    scores: List[float] = []
    for _, row in df.iterrows():
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

            # Build minimal token_probs for label_row at this position (no BOS)
            vocab_size = lps.shape[-1]
            token_probs = torch.full((1, len(window_seq), vocab_size), fill_value=-1e9, dtype=lps.dtype)
            token_probs[0, pos_rel, :] = lps

            # Compute score at the position relative to window
            score = label_row(wt, pos_rel, mt, window_seq, token_probs, tokenizer)
            total += float(score)
        scores.append(total)
    out = df.copy()
    out['delta_log_prob'] = scores
    valid = out[['delta_log_prob', 'DMS_score']].replace([np.inf, -np.inf], np.nan).dropna()
    rho = spearmanr(valid['delta_log_prob'], valid['DMS_score']).correlation if len(valid) > 1 else np.nan
    pr = pearsonr(valid['delta_log_prob'], valid['DMS_score']) if len(valid) > 1 else (np.nan, np.nan)
    r = getattr(pr, 'statistic', None)
    if r is None:
        try:
            r = pr[0]
        except Exception:
            r = np.nan
    metrics = {
        'spearman_rho': float(rho) if rho == rho else np.nan,
        'pearson_r': float(r) if r == r else np.nan,
        'n': int(len(valid)),
    }
    return out, metrics


def run_zero_shot_masked(
    dms_ids: List[str],
    model_name: str,
    mode: Optional[str] = None,
    repo_id: str = "nikraf/ProteinGym_DMS",
    results_dir: str = os.path.join('src', 'protify', 'results'),
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> pd.DataFrame:
    os.makedirs(results_dir, exist_ok=True)
    summary_records: List[Dict[str, object]] = []
    for dms_id in dms_ids:
        df = load_proteingym_dms(dms_id, mode=mode, repo_id=repo_id, hf_token=hf_token)
        if df is None or len(df) == 0:
            continue
        results_df, metrics = zero_shot_masked_scores_for_df(df, model_name, device=device)
        per_dms_path = os.path.join(results_dir, f"{dms_id}__{model_name}__zs_masked.csv")
        results_df.to_csv(per_dms_path, index=False)
        summary_records.append({'DMS_id': dms_id, 'model': model_name, 'metric': 'spearman', 'value': metrics['spearman_rho'], 'n': metrics['n']})
        summary_records.append({'DMS_id': dms_id, 'model': model_name, 'metric': 'pearson', 'value': metrics['pearson_r'], 'n': metrics['n']})
    summary_df = pd.DataFrame.from_records(summary_records)
    summary_path = os.path.join(results_dir, f"zero_shot_masked__{model_name}.csv")
    summary_df.to_csv(summary_path, index=False)
    return summary_df
