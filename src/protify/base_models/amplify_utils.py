import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union
from torch import nn
from torch import Tensor


### SwiGLU without other dependencies
class SwiGLU(nn.Module):
    """
    A Module that encapsulates the SwiGLU activation function, which combines
    linear transformations with the SiLU (Sigmoid Linear Unit) activation function.

    Args:
        in_features (int): Number of features in the input.
        hidden_features (int): Number of hidden features.
        out_features (Optional[int], optional): Number of features in the output.
            If None, it defaults to the number of input features.
        bias (bool, optional): If True, includes a bias term in the linear layers.
            Defaults to True.
        _pack_weights (bool, optional): If True, uses a single linear layer for w1 and w2.
            Defaults to True.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = True,
        *,
        _pack_weights: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self._pack_weights = _pack_weights
        self.hidden_features = hidden_features
        self.in_features = in_features
        self.out_features = out_features

        if _pack_weights:
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
            self.w1 = None
            self.w2 = None
        else:
            self.w12 = None
            self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the SwiGLU activation function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        if self._pack_weights and self.w12 is not None:
            x12 = self.w12(x)  # Shape (..., 2 * hidden_features)
            x1, x2 = x12.chunk(2, dim=-1)  # Split into two tensors along the last dimension
        else:
            assert self.w1 is not None and self.w2 is not None, "Weights w1 and w2 must be initialized."
            x1 = self.w1(x)
            x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Implements the attention mechanism in a memory-efficient way using PyTorch.
    
    Args:
        query: Tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
        key: Tensor of shape [batch_size, seq_len_k, num_heads, head_dim]
        value: Tensor of shape [batch_size, seq_len_k, num_heads, head_dim]
        attn_bias: Optional tensor to be added to attention scores, of shape 
                   [batch_size, num_heads, seq_len_q, seq_len_k]
        p: Dropout probability. Disabled if set to 0.0
        scale: Scaling factor for query @ key.transpose(). If None, defaults to 
               1 / sqrt(head_dim)
    Returns:
        Tensor of shape [batch_size, seq_len_q, num_heads, head_dim]
    """
    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attn = query @ key.transpose(-2, -1)
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    attn = F.dropout(attn, p)
    attn = attn @ value
    return attn.transpose(1, 2).contiguous()

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """

    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk) 


class ProteinTokenizer(object):
    def __init__(
        self,
        vocab_path: str,
        pad_token_id: int,
        mask_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        unk_token_id: int,
        other_special_token_ids: Optional[List[int]],
        **kwargs,
    ):
        """Vocabulary comprising the amino acids, and the special tokens <unk>, <bos>, <eos>, <pad> and <mask>.

        Args:
            vocab_path (str): Path to the vocabulary file to load.
            pad_token_id (int): <PAD> token index.
            mask_token_id (int): <MASK> token index.
            bos_token_id (int): <BOS> token index.
            eos_token_id (int): <EOS> token index.
            unk_token_id (int): <UNK> token index.
            other_special_token_ids (Optional[List[int]]): List of additional special tokens.
        """
        self._token_to_id = dict()
        self._id_to_token = dict()

        with open(vocab_path, "r") as vocab_file:
            for i, token in enumerate(vocab_file):
                token = token.strip()
                self._token_to_id[token] = i
                self._id_to_token[i] = token

        # Padding token
        self.pad_token_id = pad_token_id
        self.pad_token = self._token_to_id.get(pad_token_id)

        # Beginning and end of sequence
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.bos_token = self._token_to_id.get(bos_token_id)
        self.eos_token = self._token_to_id.get(eos_token_id)

        # Mask token
        self.mask_token_id = mask_token_id
        self.mask_token = self._token_to_id.get(mask_token_id)

        # Unknown token
        self.unk_token_id = unk_token_id
        self.unk_token = self._id_to_token.get(unk_token_id)

        # Set of all special token indices
        self.special_token_ids = set()
        self.special_token_ids.add(pad_token_id)
        self.special_token_ids.add(mask_token_id)
        self.special_token_ids.add(bos_token_id)
        self.special_token_ids.add(eos_token_id)
        self.special_token_ids.add(unk_token_id)
        if other_special_token_ids is not None:
            self.special_token_ids.update(other_special_token_ids)

    def __len__(self) -> int:
        return len(self._token_to_id)

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_token_id)

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def encode(
        self,
        tokens: List[str],
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        random_truncate: bool = True,
        **kwargs,
    ) -> Union[List[int], Tensor]:
        """Encodes a list of tokens into a list or tensor of token indices.

        Args:
            tokens (List[str]): Sequence of tokens to encode.
            max_length (Optional[int], optional): Truncate the sequence to the specified length. Defaults to None.
            add_special_tokens (bool, optional): Add special tokens <bos> and <eos> at the start and end.. Defaults to True.
            random_truncate (bool, optional): Truncate the sequence to a random subsequence of if longer than truncate.
            Defaults to True.

        Returns:
            Union[List[int], Tensor]: Token indices.
        """
        token_ids = list(map(self.token_to_id, tokens))
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        if max_length is not None and max_length < len(token_ids):
            if random_truncate:
                offset = int(torch.randint(0, len(token_ids) - max_length, (1,)).item())
            else:
                offset = 0
            token_ids = token_ids[offset : offset + max_length]
        return torch.as_tensor(token_ids, dtype=torch.long)

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Union[List[str], str]:
        """Decodes a list or tensor of token ids into a list or string of tokens.

        Args:
            token_ids (List[int]): Token indices to decode.
            skip_special_tokens (bool, optional): Skip the special tokens <bos> and <eos> at the start and end.
            Defaults to True.

        Returns:
            Union[List[str], str]: Protein.
        """
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            if len(token_ids) > 0 and token_ids[0] in self.special_token_ids:
                token_ids = token_ids[1:]
            if len(token_ids) > 0 and token_ids[-1] in self.special_token_ids:
                token_ids = token_ids[:-1]

        tokens = " ".join(map(self.id_to_token, token_ids))

        return tokens