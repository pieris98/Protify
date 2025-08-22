import re
import os
import numpy as np
import pandas as pd
from types import SimpleNamespace
from typing import List, Tuple
import torch


def _infer_model_context_len(model, tokenizer) -> int:
    """
    Infer usable context length. 
    Returns the maximum context length minus 2 for CLS and EOS tokens.
    """
    model_config = getattr(model, 'config', None)
    if model_config is None and hasattr(model, 'esm') and hasattr(model.esm, 'config'):
        model_config = model.esm.config
    max_pos = int(getattr(model_config, 'max_position_embeddings', 1024))
    num_special_tokens = getattr(tokenizer, 'num_special_tokens_to_add', lambda pair=False: tokenizer.num_special_tokens_to_add(pair))
    return max(1, max_pos - num_special_tokens)


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

def get_sequence_slices(df, target_seq, model_context_len, start_idx=1, scoring_window="optimal", indel_mode=False):
    """
    Modified from https://github.com/OATML-Markslab/Tranception/blob/2ddf40e1db9d2d180d1b5fc9d1b39ad5b04fbb6d/tranception/utils/scoring_utils.py
    Helper function that takes as input a (pandas) dataframe df that contains a list of mutant triplets (substitutions) or full mutated sequences (indels) for scoring.
    It returns a processed DMS in which sequences have been sliced to satisfy the maximum context window of the model.
    df: (dataframe) Input dataframe to be processed
    target_seq: (string) Full reference sequence (wild type) that is mutated in the DMS assay.
    model_context_len: (int) Maximum context size for the model.
    start_idx: (int) Integer to move to 0-indexing of positions (mutation triplet are typically based on 1-indexing).
    scoring_window: (string) Method to slice sequences longer than maximum context size: 
        - optimal selects a single window as large as possible via the get_optimal_window function (this is the default)
        - sliding splits the full sequence in contiguous (non-overlapping) chunks that are of size equal to the max context (except the last chunk which may be shorter)
    indel_mode: (bool) Flag to be used when scoring insertions and deletions. Otherwise assumes substitutions.
    Note: when scoring indels for sequences that would be longer than the model max context length, it is preferable to use the "sliding" scoring_window. Use "optimal" otherwise.
    """
    len_target_seq = len(target_seq)
    num_mutants = len(df['mutated_seq'])
    df=df.reset_index(drop=True)
    if scoring_window=="optimal":
        df['mutation_barycenter'] = df['mutant'].apply(lambda x: int(np.array([int(mutation[1:-1]) - start_idx for mutation in x.split(':')]).mean())) if not indel_mode else df['mutated_seq'].apply(lambda x: len(x)//2)
        df['scoring_optimal_window'] = df['mutation_barycenter'].apply(lambda x: get_optimal_window(x, len_target_seq, model_context_len)) if not indel_mode else df['mutated_seq'].apply(lambda x: (0,len(x)))
        df['sliced_mutated_seq'] = [df['mutated_seq'][index][df['scoring_optimal_window'][index][0]:df['scoring_optimal_window'][index][1]] for index in range(num_mutants)]
        df['window_start'] = df['scoring_optimal_window'].map(lambda x: x[0]) 
        df['window_end'] = df['scoring_optimal_window'].map(lambda x: x[1])
        del df['scoring_optimal_window'], df['mutation_barycenter']
        if 'mutant' in df: del df['mutant']
        df_wt=df.copy()
        df_wt['mutated_seq'] = [target_seq] * num_mutants
        if indel_mode: # For indels, we set the wild type reference to be always the same (full length) sequence. We assume here that the length is lower than model context size (otherwise "Sliding" mode should be used)
            df_wt['window_end'] = df_wt['mutated_seq'].map(lambda x:len(x))
        df_wt['sliced_mutated_seq'] = [target_seq[df_wt['window_start'][index]:df_wt['window_end'][index]] for index in range(num_mutants)]
        df = pd.concat([df,df_wt], axis=0)
        df = df.drop_duplicates()
    elif scoring_window=="sliding":
        num_windows = 1 + int( len_target_seq / model_context_len)
        df_list=[]
        start=0
        for window_index in range(1, num_windows+1):
            df_sliced = df.copy()
            df_sliced['sliced_mutated_seq'] = df_sliced['mutated_seq'].map(lambda x: x[start:start+model_context_len]) 
            df_sliced['window_start'] = [start] * num_mutants 
            df_sliced['window_end']  =  df_sliced['mutated_seq'].map(lambda x: min(len(x), start+model_context_len)) 
            df_sliced_wt = df_sliced.copy()
            df_sliced_wt['mutated_seq'] = [target_seq] * num_mutants
            df_sliced_wt['sliced_mutated_seq'] = df_sliced_wt['mutated_seq'].map(lambda x: x[start:start+model_context_len])
            df_sliced_wt['window_end'] = df_sliced_wt['mutated_seq'].map(lambda x: min(len(x), start+model_context_len)) #Need to adjust end index if WT and sequence are not same full length
            df_list.append(df_sliced)
            df_list.append(df_sliced_wt)
            start += model_context_len
        df_final = pd.concat(df_list,axis=0)
        if 'mutant' in df_final: del df_final['mutant']
        df = df_final.drop_duplicates()
    return df.reset_index(drop=True)

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

    # account for CLS at index 0
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
    """Parse ProteinGym benchmark Summary CSV and return {model_name: spearman}.

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
