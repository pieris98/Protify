import re
import os
import numpy as np
import pandas as pd
from types import SimpleNamespace
from typing import List, Tuple
import torch

def label_row(wt: str, pos: int, mt: str, sequence: str, token_probs: torch.Tensor, tokenizer) -> float:
    """
    Compute delta log-prob at a position within a sequence window.

    - wt, mt: single-letter amino acids
    - pos: index within `sequence`
    - token_probs: log-probs over vocab, shape [1, len(sequence), vocab_size]
    """
    assert 0 <= pos < len(sequence), f"Index {pos} out of range for length {len(sequence)}"
    assert sequence[pos] == wt, (
        f"WT mismatch at {pos}: seq has {sequence[pos]!r}, mutant says {wt!r}"
    )

    wt_id = tokenizer.convert_tokens_to_ids(wt)
    mt_id = tokenizer.convert_tokens_to_ids(mt)
    unk_id = getattr(tokenizer, "unk_token_id", None)

    if wt_id is None or mt_id is None or (unk_id is not None and (wt_id == unk_id or mt_id == unk_id)):
        raise ValueError(f"WT or MT is not in vocab: {wt} or {mt}")

    score = token_probs[0, pos, mt_id] - token_probs[0, pos, wt_id]
    return score.item()


def get_optimal_window(mutation_position_relative: int, seq_len_wo_special: int, model_window: int) -> list[int]:
    """
    Helper function that selects an optimal sequence window that fits the maximum model context size.
    If the sequence length is less than the maximum context size, the full sequence is returned.
    """
    half_model_window = model_window // 2
    if seq_len_wo_special <= model_window:
        return [0, seq_len_wo_special]
    elif mutation_position_relative < half_model_window:
        return [0, model_window]
    elif mutation_position_relative >= seq_len_wo_special - half_model_window:
        return [seq_len_wo_special - model_window, seq_len_wo_special]
    else:
        return [max(0, mutation_position_relative - half_model_window), min(seq_len_wo_special, mutation_position_relative + half_model_window)]


def _parse_mutant_string(mutant: str) -> List[Tuple[str, int, str]]:
    """
    Parse a ProteinGym mutant string where each mutation is separated by ':'.
    Example: "I66N:H67T:S73C" -> [("I", 65, "N"), ("H", 66, "T"), ("S", 72, "C")]
    """
    if mutant is None or (isinstance(mutant, float) and np.isnan(mutant)):
        return []
    parts = str(mutant).split(':')
    parsed: List[Tuple[str, int, str]] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r"([A-Za-z*])([0-9]+)([A-Za-z*])", p)
        if not m:
            continue
        wt, pos, mt = m.groups()
        # -1 for 0-based indexing
        parsed.append((wt, int(pos) - 1, mt))
    return parsed



@torch.no_grad()
def _masked_position_log_probs(model, tokenizer, sequence: str, pos: int, device: torch.device) -> torch.Tensor:
    """
    Return log-probs at the masked position `pos` in `sequence`.
    Handles tokenizer special tokens.
    """
    tokens = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)
    input_ids = tokens['input_ids'][0].to(device)
    attention_mask = tokens['attention_mask'][0].to(device)

    # account for BOS at index 0
    mask_pos = pos + 1
    if mask_pos <= 0 or mask_pos >= input_ids.shape[0] - 1:
        raise IndexError(f"Mask position {mask_pos} out of bounds for tokenized length {input_ids.shape[0]}")

    masked = input_ids.clone()
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = tokenizer.convert_tokens_to_ids(getattr(tokenizer, 'mask_token', '<mask>'))
    masked[mask_pos] = mask_id

    outputs = model(masked.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
    logits = outputs.logits[0, mask_pos]
    return torch.log_softmax(logits, dim=-1).detach()


@torch.no_grad()
def get_sequence_log_probability(sequence, tokenizer, model):
    """Compute the log probability of the unmasked sequence."""
    input_tokens = tokenizer.encode(sequence, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model(input_tokens)
        logits = output["logits"]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    input_ids = input_tokens[0]  # Token IDs of amino acids
    token_probs = probabilities[0, torch.arange(len(input_ids)), input_ids]

    seq_log_prob = token_probs.log().sum().item()  # Sum log probabilities

    return seq_log_prob


def collect_proteingym_spearman(args: SimpleNamespace, model_names):
    """Parse ProteinGym benchmark CSV and return {model_name: spearman}.

    Looks for Summary_performance_DMS_[substitutions|indels]_Spearman.csv and
    creates a dictionary of {model_name: spearman} for the given model names.
    
    Used in main.py to incorporate ProteinGym Spearman metrics into the main workflow.
    """
    try:
        results_root = getattr(args, 'results_dir', 'results')
        perf_out_dir = os.path.join(results_root, 'proteingym', 'benchmark_performance')
        spearman_dir = os.path.join(perf_out_dir, 'Spearman')
        sub_csv = os.path.join(spearman_dir, 'Summary_performance_DMS_substitutions_Spearman.csv')
        ind_csv = os.path.join(spearman_dir, 'Summary_performance_DMS_indels_Spearman.csv')
        csv_path = sub_csv if os.path.exists(sub_csv) else ind_csv if os.path.exists(ind_csv) else None
        if csv_path is None:
            print(f"ProteinGym Spearman summary not found in {spearman_dir}")
            return {}

        df = pd.read_csv(csv_path)
        if 'Model_name' not in df.columns or 'Average_Spearman' not in df.columns:
            print("ProteinGym summary CSV missing required columns: 'Model_name' and 'Average_Spearman'")
            return {}

        # Build lookup from Model_name -> Average_Spearman
        model_scores = {}
        for _, row in df.iterrows():
            try:
                name = str(row['Model_name'])
                score = float(row['Average_Spearman'])
            except Exception:
                continue
            model_scores[name] = score

        # Return scores for the requested model names
        out = {}
        for model_name in (model_names or []):
            if model_name in model_scores:
                out[model_name] = float(model_scores[model_name])
        return out
    except Exception as e:
        print(f"Error collecting ProteinGym Spearman metrics: {e}")
        return {}
