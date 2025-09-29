import re
import os
import numpy as np
import pandas as pd
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
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
        keep_cols = [c for c in ['mutant', 'target_seq', 'mutated_seq','window_start','window_end','sliced_mutated_seq'] if c in df.columns]
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
        keep_cols = [c for c in ['mutant', 'target_seq', 'mutated_seq','window_start','window_end','sliced_mutated_seq'] if c in df.columns]
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

def _precompute_aa_token_ids(tokenizer) -> Dict[str, int]:
    """Precompute amino acid to token ID mapping"""
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    aa_to_id = {}

    for aa in amino_acids:
        token_id = tokenizer.convert_tokens_to_ids(aa)
        if token_id is not None:
            aa_to_id[aa] = token_id

    return aa_to_id

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
def _position_log_probs(
    model,
    tokenizer,
    scoring_method: str,
    sequences: List[str],
    positions: List[int],
    device: torch.device,
    model_name: str,
    mask_token_id: Optional[int] = None,
    batch_size: int = 32,
) -> torch.Tensor:
    """Return batched log probabilities at the specified positions for each sequence.

    Parameters
    ----------
    model : pLM
    tokenizer : pLM's tokenizer
    scoring_method :
        One of {"masked_marginal", "mutant_marginal", "wildtype_marginal"}
    sequences : List[str]
        Variants to score (one sequence per position)
    positions : List[int]
        Position within each sequence
    device : torch.device
        Target device for prob tensors
    model_name : str
        Used for assertions
    mask_token_id : int
        mask token id
    batch_size : int
        Number of sequences to process at once
        
    Returns
    -------
    torch.Tensor
        Log probabilities with shape (len(sequences), vocab_size)
    """
    assert len(sequences) == len(positions), "Must have one position per sequence"
    
    all_log_probs = []
    
    # Process in batches
    for batch_start in range(0, len(sequences), batch_size):
        batch_end = min(batch_start + batch_size, len(sequences))
        batch_sequences = sequences[batch_start:batch_end]
        batch_positions = positions[batch_start:batch_end]
        
        tokens = tokenizer(
            batch_sequences,
            return_tensors='pt',
            add_special_tokens=True,
            padding=True,
        )
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        seq_lengths = attention_mask.sum(dim=1)

        if model_name not in ["GLM2-150", "GLM2-650"]:
            expected_lengths = torch.tensor([len(seq) + 2 for seq in batch_sequences], device=seq_lengths.device)
            if not torch.equal(seq_lengths, expected_lengths):
                raise AssertionError(
                    "Tokenized length must equal len(sequence)+2 for all sequences in the batch"
                )

        token_indices = torch.tensor([pos + 1 for pos in batch_positions], device=device, dtype=torch.long)
        if model_name not in ["GLM2-150", "GLM2-650"]:
            if torch.any(token_indices <= 0) or torch.any(token_indices >= (seq_lengths.to(device) - 1)):
                raise IndexError("Relative position out of bounds for tokenized sequence length")

        if scoring_method == "masked_marginal":
            mask_id = mask_token_id
            if mask_id is None:
                mask_id = tokenizer.mask_token_id
                if mask_id is None:
                    mask_id = tokenizer.convert_tokens_to_ids(getattr(tokenizer, 'mask_token', '<mask>'))
            if mask_id is None:
                raise ValueError("Tokenizer has no mask token.")
            masked_input_ids = input_ids.clone()
            batch_indices = torch.arange(masked_input_ids.size(0), device=device)
            masked_input_ids[batch_indices, token_indices] = mask_id
            outputs = model(masked_input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits.float()
        batch_indices = torch.arange(logits.size(0), device=device)
        selected_logits = logits[batch_indices, token_indices]
        batch_log_probs = torch.log_softmax(selected_logits, dim=-1)
        all_log_probs.append(batch_log_probs)
    
    # Concatenate all batches
    return torch.cat(all_log_probs, dim=0)


@torch.no_grad()
def get_sequence_log_probability(sequence, tokenizer, model, device: torch.device, model_name: str):
    """Compute the log probability of the entire, unmasked sequence using vectorized operations."""
    tokens = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)
    input_ids = tokens['input_ids'][0].to(device)
    attention_mask = tokens['attention_mask'][0].to(device)
    expected_len = len(sequence) + 2
    if model_name not in ["GLM2-150", "GLM2-650"]:
        assert input_ids.shape[0] == expected_len, (
            f"Tokenized length {input_ids.shape[0]} must equal len(sequence)+2 ({expected_len})"
        )
    
    output = model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
    logits = output["logits"]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    
    # Use torch.gather to extract log probabilities for actual tokens
    seq_start = 1
    seq_end = input_ids.size(0) - 1
    
    # Get the token IDs for the sequence positions
    token_ids = input_ids[seq_start:seq_end].unsqueeze(0).unsqueeze(-1)  # Shape: [1, seq_len, 1]
    
    # Extract log probabilities for the actual tokens
    selected_log_probs = torch.gather(log_probs[0, seq_start:seq_end], dim=-1, index=token_ids.squeeze(0))
    
    # Sum the log probabilities
    seq_log_prob = selected_log_probs.sum().item()
    return seq_log_prob

@torch.no_grad()
def calculate_pll(sequence: str, tokenizer, model, device: torch.device, model_name: str, batch_size: int = 32) -> Tuple[float, float]:
    """Calculate pseudo-log-likelihood by masking each position, with batched processing."""
    tokens = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    input_ids = tokens['input_ids'][0].to(device)
    attention_mask = tokens['attention_mask'][0].to(device)
    
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = tokenizer.convert_tokens_to_ids(getattr(tokenizer, 'mask_token', '<mask>'))
    if mask_id is None:
        raise ValueError("Tokenizer must provide a valid mask token id")
    expected_len = len(sequence) + 2
    if model_name not in ["GLM2-150", "GLM2-650"]:
        assert input_ids.shape[0] == expected_len, (
            f"Tokenized length {input_ids.shape[0]} must equal len(sequence)+2 ({expected_len})"
        )
    L = len(sequence)
    total_ll = 0.0
    
    seq_start = 1
    seq_end = input_ids.size(0) - 1
    positions = list(range(seq_start, seq_end))
    
    # Process positions in batches
    for batch_start_idx in range(0, len(positions), batch_size):
        batch_end_idx = min(batch_start_idx + batch_size, len(positions))
        batch_positions = positions[batch_start_idx:batch_end_idx]
        batch_size_len = len(batch_positions)
        
        # Create masked versions for this batch
        masked_batch = input_ids.unsqueeze(0).expand(batch_size_len, -1).clone()
        attention_mask_batch = attention_mask.unsqueeze(0).expand(batch_size_len, -1)
        
        for i, pos in enumerate(batch_positions):
            masked_batch[i, pos] = mask_id
        
        # Forward pass for the batch
        outputs = model(masked_batch, attention_mask=attention_mask_batch)
        logits = outputs.logits.float()
        
        # Extract log probabilities for the masked positions
        for i, pos in enumerate(batch_positions):
            logp = torch.log_softmax(logits[i, pos], dim=-1)
            true_id = input_ids[pos].item()
            total_ll += logp[true_id].item()
    
    return total_ll, total_ll / L

# dictionary of context lengths for supported models in Protify
MODEL_CONTEXT_LENGTH = {
    # currently supported models
    'ESM2-8': 1024,    # ESM Family models utilize huggingface.co/Synthyra/FastESM2_650
    'ESM2-35': 1024,   # style models, which have 2048 context window
    'ESM2-150': 1024,
    'ESM2-650': 1024,
    'ESM2-3B': 1024,
    'ESMC-300': 2048,
    'ESMC-600': 2048,
    'ProtBert': 1024,
    'ProtBert-BFD': 1024,
    'GLM2-150': 4096,
    'GLM2-650': 4096,
    'DSM-150': 1024,
    'DSM-650': 2048,
    'DPLM-150': 1024,
    'DPLM-650': 1024,
    'DPLM-3B': 1024,
    'Random-Transformer': 1024,
    # currently unsupported models
    'Random': None,
    'Random-ESM2-8': 2048,
    'Random-ESM2-35': 1024,
    'Random-ESM2-150': 1024,
    'Random-ESM2-650': 1024,
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
