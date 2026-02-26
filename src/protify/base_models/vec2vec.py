"""
HuggingFace-compatible vec2vec implementation for embedding translation.
Based on: "Harnessing the Universal Geometry of Embeddings" (arXiv:2505.12540)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, List
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput


# =============================================================================
# Configuration
# =============================================================================

class Vec2VecConfig(PretrainedConfig):
    """Configuration for Vec2Vec model."""
    
    model_type = "vec2vec"
    
    def __init__(
        self,
        encoder_names: List[str] = None,
        encoder_dims: List[int] = None,
        d_adapter: int = 1024,
        d_hidden: int = 1024,
        d_transform: int = 1024,
        adapter_depth: int = 3,
        transform_depth: int = 4,
        disc_dim: int = 1024,
        disc_depth: int = 5,
        weight_init: str = "kaiming",
        norm_style: str = "batch",
        normalize_embeddings: bool = True,
        # Loss coefficients
        loss_coefficient_rec: float = 1.0,
        loss_coefficient_vsp: float = 1.0,
        loss_coefficient_cc_trans: float = 10.0,
        loss_coefficient_cc_vsp: float = 10.0,
        loss_coefficient_cc_rec: float = 0.0,
        loss_coefficient_gen: float = 1.0,
        loss_coefficient_latent_gen: float = 1.0,
        loss_coefficient_similarity_gen: float = 0.0,
        loss_coefficient_disc: float = 1.0,
        loss_coefficient_r1_penalty: float = 0.0,
        # Training settings
        noise_level: float = 0.0,
        max_grad_norm: float = 1000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_names = encoder_names or ["model_a", "model_b"]
        self.encoder_dims = encoder_dims or [768, 768]
        self.d_adapter = d_adapter
        self.d_hidden = d_hidden
        self.d_transform = d_transform
        self.adapter_depth = adapter_depth
        self.transform_depth = transform_depth
        self.disc_dim = disc_dim
        self.disc_depth = disc_depth
        self.weight_init = weight_init
        self.norm_style = norm_style
        self.normalize_embeddings = normalize_embeddings
        # Loss coefficients
        self.loss_coefficient_rec = loss_coefficient_rec
        self.loss_coefficient_vsp = loss_coefficient_vsp
        self.loss_coefficient_cc_trans = loss_coefficient_cc_trans
        self.loss_coefficient_cc_vsp = loss_coefficient_cc_vsp
        self.loss_coefficient_cc_rec = loss_coefficient_cc_rec
        self.loss_coefficient_gen = loss_coefficient_gen
        self.loss_coefficient_latent_gen = loss_coefficient_latent_gen
        self.loss_coefficient_similarity_gen = loss_coefficient_similarity_gen
        self.loss_coefficient_disc = loss_coefficient_disc
        self.loss_coefficient_r1_penalty = loss_coefficient_r1_penalty
        self.noise_level = noise_level
        self.max_grad_norm = max_grad_norm

    def get_encoder_dims_dict(self) -> Dict[str, int]:
        """Return encoder dimensions as a dictionary."""
        return dict(zip(self.encoder_names, self.encoder_dims))


# =============================================================================
# Model Outputs
# =============================================================================

@dataclass
class Vec2VecOutput(ModelOutput):
    """Output type for Vec2Vec forward pass."""
    loss: Optional[torch.FloatTensor] = None
    reconstructions: Optional[Dict[str, torch.Tensor]] = None
    translations: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    latents: Optional[Dict[str, torch.Tensor]] = None
    metrics: Optional[Dict[str, float]] = None


# =============================================================================
# Model Components
# =============================================================================

def add_residual(input_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Add residual connection with dimension matching."""
    if input_x.shape[1] < x.shape[1]:
        padding = torch.zeros(x.shape[0], x.shape[1] - input_x.shape[1], device=x.device)
        input_x = torch.cat([input_x, padding], dim=1)
    elif input_x.shape[1] > x.shape[1]:
        input_x = input_x[:, :x.shape[1]]
    return x + input_x


class MLPWithResidual(nn.Module):
    """MLP with residual connections."""
    
    def __init__(
        self, 
        depth: int, 
        in_dim: int, 
        hidden_dim: int, 
        out_dim: int,
        norm_style: str = "batch",
        weight_init: str = "kaiming",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        norm_layer = nn.BatchNorm1d if norm_style == "batch" else nn.LayerNorm

        for layer_idx in range(depth):
            if layer_idx == 0:
                h_dim = out_dim if depth == 1 else hidden_dim
                self.layers.append(nn.Sequential(nn.Linear(in_dim, h_dim), nn.SiLU()))
            elif layer_idx < depth - 1:
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    norm_layer(hidden_dim),
                    nn.Dropout(p=0.1),
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(p=0.1),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, out_dim),
                ))
        self._initialize_weights(weight_init)
    
    def _initialize_weights(self, weight_init: str):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if weight_init == "kaiming":
                    nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in", nonlinearity="relu")
                elif weight_init == "xavier":
                    nn.init.xavier_normal_(module.weight)
                elif weight_init == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                module.bias.data.fill_(0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.normal_(module.bias, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            input_x = x
            x = layer(x)
            x = add_residual(input_x, x)
        return x


class Discriminator(nn.Module):
    """Discriminator network for adversarial training."""
    
    def __init__(
        self, 
        latent_dim: int, 
        hidden_dim: int = 1024, 
        depth: int = 5, 
        weight_init: str = "kaiming",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        
        if depth >= 2:
            layers = [nn.Linear(latent_dim, hidden_dim), nn.Dropout(0.0)]
            for _ in range(depth - 2):
                layers.extend([
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.0),
                ])
            layers.extend([nn.SiLU(), nn.Linear(hidden_dim, 1)])
            self.layers.append(nn.Sequential(*layers))
        else:
            self.layers.append(nn.Linear(latent_dim, 1))
        
        self._initialize_weights(weight_init)
    
    def _initialize_weights(self, weight_init: str):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if weight_init == "kaiming":
                    nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in", nonlinearity="relu")
                elif weight_init == "xavier":
                    nn.init.xavier_normal_(module.weight)
                elif weight_init == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                module.bias.data.fill_(0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# Main Model
# =============================================================================

class Vec2VecModel(PreTrainedModel):
    """
    Vec2Vec model for embedding translation between different spaces.
    
    Architecture:
        Input -> In Adapter -> Transform -> Out Adapter -> Output
    """
    
    config_class = Vec2VecConfig
    all_tied_weights_keys = {}
    
    def __init__(self, config: Vec2VecConfig):
        super().__init__(config)
        self.config = config
        encoder_dims = config.get_encoder_dims_dict()
        
        # Shared transform
        self.transform = MLPWithResidual(
            depth=config.transform_depth,
            in_dim=config.d_adapter,
            hidden_dim=config.d_transform,
            out_dim=config.d_adapter,
            norm_style=config.norm_style,
            weight_init=config.weight_init,
        )
        
        # Adapters for each encoder
        self.in_adapters = nn.ModuleDict()
        self.out_adapters = nn.ModuleDict()
        
        for name, dim in encoder_dims.items():
            self.in_adapters[name] = MLPWithResidual(
                config.adapter_depth, dim, config.d_hidden, config.d_adapter,
                config.norm_style, config.weight_init,
            )
            self.out_adapters[name] = MLPWithResidual(
                config.adapter_depth, config.d_adapter, config.d_hidden, dim,
                config.norm_style, config.weight_init,
            )
        
        # Discriminators
        self.discriminators = nn.ModuleDict()
        for name, dim in encoder_dims.items():
            self.discriminators[name] = Discriminator(
                dim, config.disc_dim, config.disc_depth, config.weight_init
            )
        self.discriminators["latent"] = Discriminator(
            config.d_adapter, config.disc_dim, config.disc_depth, config.weight_init
        )
        
        self.post_init()
    
    def add_encoder(self, name: str, dim: int, overwrite: bool = False):
        """Add a new encoder to the model."""
        if name in self.in_adapters and not overwrite:
            print(f"Encoder {name} already exists, skipping...")
            return
        
        self.in_adapters[name] = MLPWithResidual(
            self.config.adapter_depth, dim, self.config.d_hidden, self.config.d_adapter,
            self.config.norm_style, self.config.weight_init,
        )
        self.out_adapters[name] = MLPWithResidual(
            self.config.adapter_depth, self.config.d_adapter, self.config.d_hidden, dim,
            self.config.norm_style, self.config.weight_init,
        )
        self.discriminators[name] = Discriminator(
            dim, self.config.disc_dim, self.config.disc_depth, self.config.weight_init
        )
        
        # Update config
        if name not in self.config.encoder_names:
            self.config.encoder_names.append(name)
            self.config.encoder_dims.append(dim)
    
    def _get_latent(self, emb: torch.Tensor, encoder_name: str) -> torch.Tensor:
        """Get latent representation from embedding."""
        z = self.in_adapters[encoder_name](emb)
        return self.transform(z)
    
    def _decode(self, latent: torch.Tensor, encoder_name: str) -> torch.Tensor:
        """Decode latent to target embedding space."""
        out = self.out_adapters[encoder_name](latent)
        if self.config.normalize_embeddings:
            out = F.normalize(out, p=2, dim=1)
        return out
    
    def translate(self, embeddings: torch.Tensor, src: str, tgt: str) -> torch.Tensor:
        """Translate embeddings from source to target space."""
        latent = self._get_latent(embeddings, src)
        return self._decode(latent, tgt)
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        noise_level: float = None,
        return_latents: bool = False,
    ) -> Vec2VecOutput:
        """
        Forward pass computing reconstructions and translations.
        
        Args:
            inputs: Dict mapping encoder names to embeddings
            noise_level: Optional noise for training
            return_latents: Whether to return latent representations
        """
        noise_level = noise_level if noise_level is not None else self.config.noise_level
        
        reconstructions = {}
        translations = {}
        latents = {}
        
        for src_name, emb in inputs.items():
            # Add noise during training
            if self.training and noise_level > 0.0:
                emb = emb + torch.randn_like(emb) * noise_level
                emb = F.normalize(emb, p=2, dim=1)
            
            latent = self._get_latent(emb, src_name)
            if return_latents:
                latents[src_name] = latent
            
            for tgt_name in inputs.keys():
                decoded = self._decode(latent, tgt_name)
                if tgt_name == src_name:
                    reconstructions[src_name] = decoded
                else:
                    if tgt_name not in translations:
                        translations[tgt_name] = {}
                    translations[tgt_name][src_name] = decoded
        
        return Vec2VecOutput(
            reconstructions=reconstructions,
            translations=translations,
            latents=latents if return_latents else None,
        )


# =============================================================================
# Loss Functions
# =============================================================================

def reconstruction_loss(inputs: Dict[str, torch.Tensor], recons: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Reconstruction loss (1 - cosine similarity)."""
    loss = sum(1 - F.cosine_similarity(inputs[k], recons[k], dim=1).mean() for k in inputs)
    return loss / len(inputs)


def translation_loss(inputs: Dict[str, torch.Tensor], translations: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Translation loss (1 - cosine similarity)."""
    loss = 0.0
    count = 0
    for tgt, emb in inputs.items():
        for trans in translations[tgt].values():
            loss += 1 - F.cosine_similarity(emb, trans, dim=1).mean()
            count += 1
    return loss / max(count, 1)


def vsp_loss(inputs: Dict[str, torch.Tensor], translations: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Vector Space Preservation (VSP) loss."""
    loss = 0.0
    count = 0
    EPS = 1e-10
    
    for out_name in inputs:
        for in_name in translations[out_name]:
            B = F.normalize(inputs[out_name].detach(), p=2, dim=1)
            A = F.normalize(translations[out_name][in_name], p=2, dim=1)
            
            in_sims = B @ B.T
            out_sims = A @ A.T
            out_sims_reflected = A @ B.T
            
            loss += (in_sims - out_sims).abs().mean()
            loss += (in_sims - out_sims_reflected).abs().mean()
            count += 1
    
    return loss / max(count, 1)


from typing import Optional, Union, List, Dict
from transformers import AutoModel, AutoTokenizer
from .base_tokenizer import BaseSequenceTokenizer
from .supported_models import all_presets_with_paths

from pooler import Pooler


presets = {
    'vec2vec-ESM2-8-ESM2-35': 'Synthyra/ESM2-8-ESM2-35-sequence-sequence',
    'vec2vec-ESM2-8-ESM2-150': 'Synthyra/ESM2-8-ESM2-150-sequence-sequence',
    'vec2vec-ESM2-8-ESM2-650': 'Synthyra/ESM2-8-ESM2-650-sequence-sequence',
    'vec2vec-ESM2-8-ESM2-3B': 'Synthyra/ESM2-8-ESM2-3B-sequence-sequence',
    'vec2vec-ESM2-35-ESM2-150': 'Synthyra/ESM2-35-ESM2-150-sequence-sequence',
    'vec2vec-ESM2-35-ESM2-650': 'Synthyra/ESM2-35-ESM2-650-sequence-sequence',
    'vec2vec-ESM2-35-ESM2-3B': 'Synthyra/ESM2-35-ESM2-3B-sequence-sequence',
    'vec2vec-ESM2-150-ESM2-650': 'Synthyra/ESM2-150-ESM2-650-sequence-sequence',
    'vec2vec-ESM2-150-ESM2-3B': 'Synthyra/ESM2-150-ESM2-3B-sequence-sequence',
    'vec2vec-ESM2-650-ESM2-3B': 'Synthyra/ESM2-650-ESM2-3B-sequence-sequence',
}


class Vec2VecTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'longest')
        kwargs.setdefault('add_special_tokens', True)
        tokenized = self.tokenizer(sequences, **kwargs)
        return tokenized


class Vec2VecForEmbedding(nn.Module):
    def __init__(
        self,
        config: Vec2VecConfig,
        base_model: AutoModel,
        vec2vec_model: Vec2VecModel,
        model_name_a: str,
        model_name_b: str,
    ):
        super().__init__()
        self.base_model = base_model
        self.vec2vec_model = vec2vec_model
        self.config = config
        self.pooler = Pooler(['mean', 'var'])
        self.model_name_a = model_name_a
        self.model_name_b = model_name_b
        self.normalize = config.normalize_embeddings

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = False,
            **kwargs,
    ) -> torch.Tensor:
        # only vector embeddings, don't use output_attentions, etc.
        base_state = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        base_vec = self.pooler(base_state, attention_mask=attention_mask)
        if self.normalize:
            base_vec = F.normalize(base_vec, p=2, dim=1)
        translated_ab = self.vec2vec_model.translate(base_vec, src=self.model_name_a, tgt=self.model_name_b)
        return translated_ab


def get_vec2vec_tokenizer(preset: str):
    # TODO work with new Vec2Vec .tokenizer_a and .tokenizer_b
    try:
        tokenizer = AutoTokenizer.from_pretrained(all_presets_with_paths[preset], trust_remote_code=True)
    except:
        model = AutoModel.from_pretrained(all_presets_with_paths[preset], trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model.config.tokenizer_name)
    return Vec2VecTokenizerWrapper(tokenizer)


def build_vec2vec_model(preset: str, masked_lm: bool = False, dtype: torch.dtype = None, **kwargs):
    if masked_lm:
        raise ValueError("Masked LM is not supported for Vec2VecForEmbedding")
    else:
        model_path = presets[preset]
        config = Vec2VecConfig.from_pretrained(model_path)
        encoder_names = config.encoder_names
        encoder_dims = config.encoder_dims

        if encoder_dims[0] >= encoder_dims[1]:
            model_name_a = encoder_names[0]
            model_name_b = encoder_names[1]
        else:
            model_name_a = encoder_names[1]
            model_name_b = encoder_names[0]

        base_model = AutoModel.from_pretrained(all_presets_with_paths[model_name_a], dtype=dtype, trust_remote_code=True)
        base_tokenizer = base_model.tokenizer
        vec2vec_model = Vec2VecModel(config).from_pretrained(model_path)
        model = Vec2VecForEmbedding(config, base_model, vec2vec_model, model_name_a, model_name_b)
        tokenizer = Vec2VecTokenizerWrapper(base_tokenizer)
        return model, tokenizer


def get_vec2vec_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    raise ValueError("Vec2VecForTraining is not supported yet")


if __name__ == '__main__':
    # py -m src.protify.base_models.vec2vec
    model, tokenizer = build_vec2vec_model('ESM2-8-ESM2-35')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
