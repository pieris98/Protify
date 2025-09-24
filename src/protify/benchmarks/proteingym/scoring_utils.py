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
    assert token_probs.shape[1] == len(sequence), (
        f"token_probs length {token_probs.shape[1]} must match sequence length {len(sequence)}"
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
        df_wt=df.copy()
        df_wt['mutated_seq'] = [target_seq] * num_mutants
        if indel_mode: # For indels, we set the wild type reference to be always the same (full length) sequence. We assume here that the length is lower than model context size (otherwise "Sliding" mode should be used)
            df_wt['window_end'] = df_wt['mutated_seq'].map(lambda x:len(x))
        df_wt['sliced_mutated_seq'] = [target_seq[df_wt['window_start'][index]:df_wt['window_end'][index]] for index in range(num_mutants)]
        df = pd.concat([df,df_wt], axis=0)
        df = df.drop_duplicates()
        # Keep only cols needed downstream
        keep_cols = [c for c in ['mutant', 'target_seq', 'mutated_seq','window_start','window_end','sliced_mutated_seq']]
        df = df[keep_cols]
    elif scoring_window=="sliding":
        if model_context_len is None:
            model_context_len = len_target_seq
        df_list=[]
        start=0
        while start < len_target_seq:
            end = min(start + model_context_len, len_target_seq)
            df_sliced = df.copy()
            df_sliced['sliced_mutated_seq'] = df_sliced['mutated_seq'].map(lambda x: x[start:end]) 
            df_sliced['window_start'] = [start] * num_mutants 
            df_sliced['window_end']  =  df_sliced['mutated_seq'].map(lambda x: (min(len(x), end))) 
            df_sliced_wt = df_sliced.copy()
            df_sliced_wt['mutated_seq'] = [target_seq] * num_mutants
            df_sliced_wt['sliced_mutated_seq'] = df_sliced_wt['mutated_seq'].map(lambda x: x[start:end])
            df_sliced_wt['window_end'] = df_sliced_wt['mutated_seq'].map(lambda x: min(len(x), end))
            df_list.append(df_sliced)
            df_list.append(df_sliced_wt)
            start = end
        df_final = pd.concat(df_list,axis=0)
        df = df_final.drop_duplicates()
        # Keep only cols needed downstream
        keep_cols = [c for c in ['mutant', 'target_seq', 'mutated_seq','window_start','window_end','sliced_mutated_seq']]
        df = df[keep_cols]
    return df.reset_index(drop=True)

def _apply_mutations_to_sequence(wt_seq: str, mutations: List[Tuple[str, int, str]]) -> str:
    """Apply mutations to wildtype sequence."""
    mutant_seq = list(wt_seq)
    for wt, pos, mt in mutations:
        assert 0 <= pos < len(mutant_seq), f"Mutation position {pos} out of range [0, {len(mutant_seq)})"
        if mutant_seq[pos] != wt:
            raise ValueError(f"WT mismatch at pos {pos}: expected {wt}, found {mutant_seq[pos]}")
        mutant_seq[pos] = mt
    return ''.join(mutant_seq)
    
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


def collect_proteingym_spearman(args: SimpleNamespace, model_names):
    """Parse ProteinGym benchmark Summary CSV and return {model_name: spearman}.

    Looks for Summary_performance_DMS_[substitutions|indels]_Spearman.csv and
    creates a dictionary of {model_name: spearman} for the given model names.
    
    Used in main.py to incorporate ProteinGym Spearman metrics into the main workflow.
    """
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


'''
Scoring Functions
'''

@torch.no_grad()
def _position_log_probs(model, tokenizer, scoring_method: str, sequence: str, pos: int, device: torch.device, model_name: str) -> torch.Tensor:
    """
    Return log-probs at the target position `pos`.
    We tokenize with special tokens and map to the corresponding token index (pos + 1).
    For masked_marginal, we replace w/ mask token before forward.
    """
    tokens = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)
    input_ids = tokens['input_ids'][0].to(device)
    attention_mask = tokens['attention_mask'][0].to(device)
    extra_leading_special = 1 if model_name in ("GLM2-150", "GLM2-650") else 0
    expected_len = len(sequence) + 2 + extra_leading_special
    assert input_ids.shape[0] == expected_len, (
        f"Tokenized length {input_ids.shape[0]} must equal len(sequence)+2+extra ({expected_len})"
    )
    token_idx = pos + 1 + extra_leading_special
    
    if token_idx <= 0 or token_idx >= input_ids.shape[0] - 1:
        raise IndexError(f"Position {token_idx} out of bounds for tokenized length {input_ids.shape[0]}")
    
    if scoring_method == "masked_marginal":
        mask_id = tokenizer.mask_token_id
        if mask_id is None:
            mask_id = tokenizer.convert_tokens_to_ids(getattr(tokenizer, 'mask_token', '<mask>'))
        if mask_id is None:
            raise ValueError("Tokenizer has no mask token.")
        masked = input_ids.clone()
        masked[token_idx] = mask_id
        outputs = model(masked.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
    else:
        outputs = model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
    logits = outputs.logits[0, token_idx]
    assert logits.dim() == 1, f"Expected 1D logits for a single position, got shape {tuple(logits.shape)}"
    lps = torch.log_softmax(logits.float(), dim=-1)
    return lps.detach().cpu()


@torch.no_grad()
def get_sequence_log_probability(sequence, tokenizer, model, device: torch.device, model_name: str):
    """Compute the log probability of the entire, unmasked sequence."""
    tokens = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)
    input_ids = tokens['input_ids'][0].to(device)
    attention_mask = tokens['attention_mask'][0].to(device)
    extra_leading_special = 1 if model_name in ("GLM2-150", "GLM2-650") else 0
    expected_len = len(sequence) + 2 + extra_leading_special
    assert input_ids.shape[0] == expected_len, (
        f"Tokenized length {input_ids.shape[0]} must equal len(sequence)+2 ({expected_len})"
    )
    with torch.no_grad():
        output = model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        logits = output["logits"]
        probabilities = torch.nn.functional.softmax(logits.float(), dim=-1)
    seq_log_prob = 0.0
    seq_start = 1 + extra_leading_special
    seq_end = input_ids.size(0) - 1
    for pos in range(seq_start, seq_end):
        token_id = input_ids[pos].item()
        token_prob = probabilities[0, pos, token_id]
        seq_log_prob += token_prob.log().item()
    return seq_log_prob

@torch.no_grad()
def calculate_pll(sequence: str, tokenizer, model, device: torch.device, model_name: str) -> Tuple[float, float]:
    """Calculate pseudo-log-likelihood by masking each position iteratively."""
    tokens = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    input_ids = tokens['input_ids'][0].to(device)
    attention_mask = tokens['attention_mask'][0].to(device)
    
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = tokenizer.convert_tokens_to_ids(getattr(tokenizer, 'mask_token', '<mask>'))
    if mask_id is None:
        raise ValueError("Tokenizer must provide a valid mask token id")
    extra_leading_special = 1 if model_name in ("GLM2-150", "GLM2-650") else 0
    expected_len = len(sequence) + 2 + extra_leading_special
    assert input_ids.shape[0] == expected_len, (
        f"Tokenized length {input_ids.shape[0]} must equal len(sequence)+2+extra ({expected_len})"
    )
    L = len(sequence)
    total_ll = 0.0
    with torch.no_grad():
        seq_start = 1 + extra_leading_special
        seq_end = input_ids.size(0) - 1
        for pos in range(seq_start, seq_end):
            masked = input_ids.clone()
            masked[pos] = mask_id
            outputs = model(masked.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            logits = outputs.logits[0, pos]
            logp = torch.log_softmax(logits.float(), -1)
            true_id = input_ids[pos].item()
            total_ll += logp[true_id].item()
    return total_ll, total_ll / L

# dictionary of context lengths for supported models in Protify
MODEL_CONTEXT_LENGTH = {
    # currently supported models
    'ESM2-8': 2048,    # ESM Family models utilize huggingface.co/Synthyra/FastESM2_650
    'ESM2-35': 2048,   # style models, which have 2048 context window
    'ESM2-150': 2048,
    'ESM2-650': 2048,
    'ESM2-3B': 2048,
    'ESMC-300': 2048,
    'ESMC-600': 2048,
    'ProtBert': 40000, 
    'ProtBert-BFD': 40000,
    'GLM2-150': 4096,
    'GLM2-650': 4096,
    'DSM-150': 1024,
    'DSM-650': 2048,
    'DPLM-150': 1024,
    'DPLM-650': 1024,
    'DPLM-3B': 1024,
    # currently unsupported models
    'Random': None,
    'Random-Transformer': 2048,
    'Random-ESM2-8': 2048,
    'Random-ESM2-35': 2048,
    'Random-ESM2-150': 2048,
    'Random-ESM2-650': 2048,
    'ESM2-diff-150': 1026,
    'ESM2-diffAV-150': 1026,
    'ProtT5': 512,
    'ProtT5-XL-UniRef50-full-prec': 512,
    'ProtT5-XXL-UniRef50': 512,
    'ProtT5-XL-BFD': 512,
    'ProtT5-XXL-BFD': 512,
    'ANKH-Base': 512,
    'ANKH-Large': 512,
    'ANKH2-Large': 512,
    'GLM2-GAIA': 4096,
    'DSM-PPI': 1024,
    'ProtCLM-1b': 2048,
    'OneHot-Protein': None,
    'OneHot-DNA': None,
    'OneHot-RNA': None,
    'OneHot-Codon': None,
}
