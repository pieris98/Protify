# Model Components

This page is for developers who need to understand or extend the shared neural building blocks used by transformer-style probes. It summarizes [attention.py](../src/protify/model_components/attention.py), [attention_utils.py](../src/protify/model_components/attention_utils.py), [transformer.py](../src/protify/model_components/transformer.py), and [mlp.py](../src/protify/model_components/mlp.py), and which probes use them.

---

## Overview

The `model_components` package provides:

- **Attention:** Rotary embeddings, multi-head attention with pluggable backends (flex, sdpa, kernels), and attention-based logits for interpnet.
- **Transformer:** A stack of transformer blocks (attention + SwiGLU FFN) used by the transformer probe and by InterpNet.
- **MLP:** SwiGLU and feed-forward helpers used in transformer blocks and in the linear probe (intermediate sizing).

Probes that use these components:

- **Transformer probe:** Uses `Transformer`, `LayerNorm`, and `intermediate_correction_fn` (from mlp).
- **InterpNet probe:** Uses `Transformer`, `AttentionLogitsSequence`, `AttentionLogitsToken`, and `Linear` (from attention).
- **Linear probe:** Uses only `intermediate_correction_fn` from mlp (for classifier sizing).

---

## attention.py

- **rotate_half(x, interleaved):** Rotates half the dimensions for rotary embeddings.
- **apply_rotary_emb_torch(x, cos, sin, interleaved):** Applies rotary embeddings to a tensor.
- **RotaryEmbedding(dim, base, interleaved, scaling_factor):** Module that caches cos/sin for a max sequence length.
- **MultiHeadAttention(hidden_size, n_heads, rotary, attention_backend, use_bias):** Multi-head attention with optional rotary. Dispatches to flex_attention_func, kernels_attention_func, or sdpa_attention_func from attention_utils. Supports 2D and 4D attention masks and optional BlockMask for flex. Can return attention weights and s_max.
- **AttentionLogitsSequence,** **AttentionLogitsToken:** Used by InterpNet for sequence-level and token-level classification from attention over learned parameters.
- **Linear,** **LayerNorm:** Re-exports of `nn.Linear` and `nn.LayerNorm` for use across probes.

---

## attention_utils.py

- **AttentionBackend:** Enum (KERNELS, FLEX, SDPA) for the attention implementation.
- **resolve_attention_backend(name):** Returns the backend to use.
- **_repeat_kv(keys, values, n_rep):** For grouped-query style repetition.
- **sdpa_attention_func(...):** PyTorch SDPA-based attention.
- **flex_attention_func(...):** Flex attention (e.g. torch.nn.functional.flex_attention when available).
- **kernels_attention_func(...):** Custom kernel-based attention path.
- **build_attention_masks(...):** Builds 2D and 4D masks (and BlockMask for flex) from attention_mask.
- **pad_input,** **_unpad_input:** Padding and unpadding for block-sparse or kernel paths.
- **IndexFirstAxis,** **IndexPutFirstAxis:** Helpers for indexing in attention kernels.

These are used by `MultiHeadAttention` and by the transformer block when calling the chosen attention function.

---

## transformer.py

- **TransformerBlock(hidden_size, n_heads, expansion_ratio, dropout, rotary, use_bias, attention_backend):** Single block: LayerNorm, MultiHeadAttention, residual, then SwiGLU FFN (`swiglu_ln_ffn` from mlp).
- **TransformerOutput:** Dataclass extending ModelOutput (loss, logits, last_hidden_state, hidden_states, attentions, s_max).
- **Transformer(hidden_size, n_heads, n_layers, expansion_ratio, dropout, rotary, use_bias, attention_backend):** Stack of TransformerBlocks. Forward accepts hidden_states, attention_mask_2d/4d, flex_block_mask, output_attentions, output_s_max.
- **TransformerForMaskedLM:** Wrapper for masked LM; not used by the standard probes.

The **transformer probe** and **InterpNet** both use `Transformer` as the backbone above embeddings; InterpNet then applies `AttentionLogitsSequence` or `AttentionLogitsToken` instead of a simple pooler + linear classifier.

---

## mlp.py

- **intermediate_correction_fn(hidden_size, classifier_size):** Returns an intermediate dimension (used for classifier sizing in linear and transformer probes).
- **SwiGLU:** Swish-gated linear unit module.
- **swiglu_ln_ffn(hidden_size, expansion_ratio, dropout, use_bias):** LayerNorm + SwiGLU feed-forward used in TransformerBlock.

---

## Dependency summary

| Component | Used by |
|-----------|---------|
| Transformer, LayerNorm, swiglu_ln_ffn | Transformer probe, InterpNet |
| MultiHeadAttention, RotaryEmbedding, build_attention_masks, attention backends | Transformer block (hence transformer probe, InterpNet) |
| AttentionLogitsSequence, AttentionLogitsToken | InterpNet only |
| intermediate_correction_fn | Linear probe, transformer probe (classifier sizing) |

---

## See also

- [Probes and training](probes_and_training.md) for probe types and training flows
- [Configuration](cli_and_config.md) for attention_backend and related CLI options
