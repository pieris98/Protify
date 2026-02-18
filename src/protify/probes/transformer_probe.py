import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from typing import List, Optional


try:
    from ..pooler import Pooler
except ImportError:
    try:
        from protify.pooler import Pooler
    except ImportError:
        from pooler import Pooler
try:
    from ..model_components.mlp import intermediate_correction_fn
except ImportError:
    try:
        from protify.model_components.mlp import intermediate_correction_fn
    except ImportError:
        from model_components.mlp import intermediate_correction_fn
try:
    from ..model_components.transformer import Transformer, TokenFormer
except ImportError:
    try:
        from protify.model_components.transformer import Transformer, TokenFormer
    except ImportError:
        from model_components.transformer import Transformer, TokenFormer
from .losses import get_loss_fct


class TransformerProbeConfig(PretrainedConfig):
    model_type = "probe"
    def __init__(
            self,
            input_size: int = 768,
            hidden_size: int = 512,
            classifier_size: int = 4096,
            transformer_dropout: float = 0.1,
            classifier_dropout: float = 0.2,
            num_labels: int = 2,
            n_layers: int = 1,
            token_attention: bool = False,
            n_heads: int = 4,
            task_type: str = 'singlelabel',
            rotary: bool = True,
            pre_ln: bool = True,
            probe_pooling_types: List[str] = ['mean', 'cls'],
            use_token_type_ids: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.classifier_size = classifier_size
        self.transformer_dropout = transformer_dropout
        self.classifier_dropout = classifier_dropout
        self.task_type = task_type
        self.num_labels = num_labels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.rotary = rotary
        self.pre_ln = pre_ln
        self.pooling_types = probe_pooling_types
        self.token_attention = token_attention
        self.use_token_type_ids = use_token_type_ids


class TransformerForSequenceClassification(PreTrainedModel):
    config_class = TransformerProbeConfig
    all_tied_weights_keys = {}
    def __init__(self, config: TransformerProbeConfig):
        super().__init__(config)
        self.config = config
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels
        self.input_size = config.input_size
        self.use_token_type_ids = getattr(config, 'use_token_type_ids', False)

        if config.pre_ln:
            self.input_layer = nn.Sequential(
                nn.LayerNorm(config.input_size),
                nn.Linear(config.input_size, config.hidden_size)
            )
        else:
            self.input_layer = nn.Linear(config.input_size, config.hidden_size)

        # Learned token type embeddings (e.g. protein A vs protein B for PPI tasks):
        # type 0 = protein A, type 1 = protein B
        # Gives the model an explicit signal for which tokens belong to which segment
        if self.use_token_type_ids:
            self.token_type_embedding = nn.Embedding(2, config.hidden_size)

        transformer_class = TokenFormer if config.token_attention else Transformer
        self.transformer = transformer_class(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            expansion_ratio=8/3,
            dropout=config.transformer_dropout,
            rotary=True,
        )

        classifier_input_size = config.hidden_size * len(config.pooling_types)
        proj_dim = intermediate_correction_fn(expansion_ratio=2, hidden_size=config.num_labels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_size),
            nn.Linear(classifier_input_size, config.classifier_size),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_size, proj_dim),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(proj_dim, config.num_labels)
        )
        self.pooler = Pooler(config.pooling_types)

    def forward(
            self,
            embeddings,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> SequenceClassifierOutput:
        # Convert embeddings to match model's dtype to avoid dtype mismatch errors
        # This handles cases where embeddings are fp32 but model is fp16 (or vice versa)
        embeddings = embeddings.to(next(self.input_layer.parameters()).dtype)
        x = self.input_layer(embeddings)

        # Add token type embeddings to break A/B symmetry (e.g. for PPI tasks)
        if self.use_token_type_ids and token_type_ids is not None:
            x = x + self.token_type_embedding(token_type_ids)

        x = self.transformer(x, attention_mask)
        x = self.pooler(x, attention_mask)
        logits = self.classifier(x)
        if self.task_type == 'sigmoid_regression':
            logits = logits.sigmoid()
        loss = None
        if labels is not None:
            if self.task_type == 'regression':
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.task_type == 'sigmoid_regression':
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.task_type == 'multilabel':
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
    

class TransformerForTokenClassification(PreTrainedModel):
    config_class = TransformerProbeConfig
    all_tied_weights_keys = {}
    def __init__(self, config: TransformerProbeConfig):
        super().__init__(config)
        self.config = config
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels
        self.input_size = config.input_size
        self.input_layer = nn.Linear(config.input_size, config.hidden_size)

        transformer_class = TokenFormer if config.token_attention else Transformer
        self.transformer = transformer_class(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            expansion_ratio=8/3,
            dropout=config.transformer_dropout,
            rotary=True,
        )

        proj_dim = intermediate_correction_fn(expansion_ratio=2, hidden_size=config.num_labels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.classifier_size),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_size, proj_dim),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, config.num_labels)
        )

    def forward(
            self,
            embeddings,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> TokenClassifierOutput:
        # Convert embeddings to match model's dtype to avoid dtype mismatch errors
        # This handles cases where embeddings are fp32 but model is fp16 (or vice versa)
        embeddings = embeddings.to(next(self.input_layer.parameters()).dtype)
        x = self.input_layer(embeddings)
        x = self.transformer(x, attention_mask)
        logits = self.classifier(x)
        if self.task_type == 'sigmoid_regression':
            logits = logits.sigmoid()
        loss = None
        if labels is not None:
            if self.task_type == 'regression':
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.task_type == 'sigmoid_regression':
                
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.task_type == 'multilabel':
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
