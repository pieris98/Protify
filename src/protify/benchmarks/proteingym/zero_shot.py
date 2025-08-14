import os
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional, Tuple
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from tqdm.auto import tqdm
from base_models.get_base_models import get_base_model
from .data_loader import load_proteingym_dms
from .scoring_utils import (
    label_row,
    get_optimal_window,
    _parse_mutant_string,
    _masked_position_log_probs,
)



def zero_shot_masked_scores_for_df(
    df: pd.DataFrame,
    model_name: str,
    device: Optional[str] = None,
    progress: bool = True,
    tqdm_position: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
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

    # Cache masked-position log-probs by (window_seq, idx_rel_0)
    log_prob_cache: Dict[Tuple[str, int], torch.Tensor] = {}

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

    # Classification-style metrics
    auc = np.nan
    mcc = np.nan
    top10p_recall = np.nan
    if len(valid) > 1:
        y_scores = valid['delta_log_prob'].to_numpy(dtype=float)
        y_true_reg = valid['DMS_score'].to_numpy(dtype=float)
        try:
            # For AUC/MCC: define positives via median split of DMS_score
            thr_cls = float(np.quantile(y_true_reg, 0.5))
            y_true_cls = (y_true_reg >= thr_cls).astype(int)
            has_pos = int(y_true_cls.sum()) > 0
            has_neg = int((1 - y_true_cls).sum()) > 0
            if has_pos and has_neg:
                # AUC
                try:
                    auc = float(roc_auc_score(y_true_cls, y_scores))
                except Exception:
                    auc = np.nan
                # Best-threshold MCC
                try:
                    thresholds = np.unique(y_scores)
                    if thresholds.size > 512:
                        idx = np.linspace(0, thresholds.size - 1, num=512, dtype=int)
                        thresholds = thresholds[idx]
                    best_mcc = -np.inf
                    for t in thresholds:
                        y_pred = (y_scores >= t).astype(int)
                        val = matthews_corrcoef(y_true_cls, y_pred)
                        if val > best_mcc:
                            best_mcc = val
                    mcc = float(best_mcc)
                except Exception:
                    mcc = np.nan
            # Top-10% recall@k (k = ceil(0.1 * n)) using top-decile by DMS_score as positives
            try:
                thr_top = float(np.quantile(y_true_reg, 0.9))
                y_true_top = (y_true_reg >= thr_top).astype(int)
                k = max(1, int(np.ceil(0.10 * len(y_scores))))
                top_idx = np.argpartition(y_scores, -k)[-k:]
                tp_at_k = int(y_true_top[top_idx].sum())
                pos = int(y_true_top.sum())
                top10p_recall = float(tp_at_k / pos) if pos > 0 else np.nan
            except Exception:
                top10p_recall = np.nan
        except Exception:
            pass
    metrics = {
        'spearman_rho': float(rho) if rho == rho else np.nan,
        'auc': float(auc) if auc == auc else np.nan,
        'mcc': float(mcc) if mcc == mcc else np.nan,
        'top10p_recall': float(top10p_recall) if top10p_recall == top10p_recall else np.nan,
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
    show_progress: bool = True,
) -> pd.DataFrame:
    os.makedirs(results_dir, exist_ok=True)
    summary_records: List[Dict[str, object]] = []
    assay_iterator = dms_ids
    if show_progress:
        assay_iterator = tqdm(dms_ids, desc="All assays", unit="assay", position=0)
    for dms_id in assay_iterator:
        df = load_proteingym_dms(dms_id, mode=mode, repo_id=repo_id, hf_token=hf_token)
        if df is None or len(df) == 0:
            continue
        if show_progress and hasattr(assay_iterator, 'set_description_str'):
            assay_iterator.set_description_str(f"Assay {dms_id}")
        results_df, metrics = zero_shot_masked_scores_for_df(
            df,
            model_name,
            device=device,
            progress=show_progress,
            tqdm_position=1,
        )
        per_dms_path = os.path.join(results_dir, f"{dms_id}__{model_name}__zs_masked.csv")
        # Only keep one row of 'target_seq' (present in first row), blank elsewhere
        results_to_save = results_df.copy()
        if 'target_seq' in results_to_save.columns and len(results_to_save) > 1:
            results_to_save.loc[1:, 'target_seq'] = ''
        results_to_save.to_csv(per_dms_path, index=False)
        summary_records.append({'DMS_id': dms_id, 'model': model_name, 'metric': 'spearman', 'value': metrics['spearman_rho'], 'n': metrics['n']})
        summary_records.append({'DMS_id': dms_id, 'model': model_name, 'metric': 'auc', 'value': metrics['auc'], 'n': metrics['n']})
        summary_records.append({'DMS_id': dms_id, 'model': model_name, 'metric': 'mcc', 'value': metrics['mcc'], 'n': metrics['n']})
        summary_records.append({'DMS_id': dms_id, 'model': model_name, 'metric': 'top10p_recall', 'value': metrics['top10p_recall'], 'n': metrics['n']})
        # Print per-assay results on the fly
        tqdm.write(
            f"[Assay {dms_id}] spearman_rho={metrics['spearman_rho']:.4f}, "
            f"auc={metrics['auc']:.4f}, mcc={metrics['mcc']:.4f}, top10%_recall={metrics['top10p_recall']:.4f}, "
            f"n={metrics['n']} → saved: {per_dms_path}"
        )
    summary_df = pd.DataFrame.from_records(summary_records)
    # Append summary rows: mean per metric, and total n per metric
    metrics_to_summarize = ['spearman', 'auc', 'mcc', 'top10p_recall']
    for m in metrics_to_summarize:
        sub = summary_df[summary_df['metric'] == m]
        if len(sub) == 0:
            continue
        mean_val = float(sub['value'].mean())
        total_n = int(sub['n'].sum())
        summary_df = pd.concat([
            summary_df,
            pd.DataFrame.from_records([{ 'DMS_id': 'Summary', 'model': model_name, 'metric': m, 'value': mean_val, 'n': total_n }])
        ], ignore_index=True)
    summary_path = os.path.join(results_dir, f"zero_shot_masked__{model_name}.csv")
    summary_df.to_csv(summary_path, index=False)
    # Print summary metrics at the very end
    try:
        lines = []
        for m in metrics_to_summarize:
            row = summary_df[(summary_df['DMS_id'] == 'Summary') & (summary_df['metric'] == m)]
            if len(row) == 1:
                val = row['value'].iloc[0]
                n_total = row['n'].iloc[0]
                pretty = 'top10% recall' if m == 'top10p_recall' else m
                lines.append(f"{pretty}: {val:.4f} (n={n_total})")
        if lines:
            tqdm.write("Summary metrics → " + "; ".join(lines))
    except Exception:
        pass
    tqdm.write(f"Saved summary: {summary_path}")
    return summary_df
