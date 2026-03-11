import torch
from dataclasses import dataclass
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional


try:
    from ..model_components.attention import AttentionLogitsSequence, AttentionLogitsToken, Linear
except ImportError:
    try:
        from protify.model_components.attention import AttentionLogitsSequence, AttentionLogitsToken, Linear
    except ImportError:
        from model_components.attention import AttentionLogitsSequence, AttentionLogitsToken, Linear

try:
    from ..model_components.transformer import Transformer
except ImportError:
    try:
        from protify.model_components.transformer import Transformer
    except ImportError:
        from model_components.transformer import Transformer

from .losses import get_loss_fct


@dataclass
class InterpNetOutput(SequenceClassifierOutput):
    s_max: tuple[list[torch.Tensor] | None, ...] | None = None


class InterpNetConfig(PretrainedConfig):
    model_type = "interpnet"

    def __init__(
        self,
        input_size: int = 768,
        hidden_size: int = 512,
        dropout: float = 0.2,
        num_labels: int = 2,
        n_layers: int = 1,
        sim_type: str = "dot",
        n_heads: int = 4,
        task_type: str = "singlelabel",
        expansion_ratio: float = 8 / 3,
        rotary: bool = True,
        attention_backend: str = "flex",
        output_s_max: bool = False,
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert task_type != "regression" or num_labels == 1, "Regression task must have exactly one label."
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.task_type = task_type
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.sim_type = sim_type
        self.expansion_ratio = expansion_ratio
        self.n_heads = n_heads
        self.rotary = rotary
        self.attention_backend = attention_backend
        self.output_s_max = output_s_max
        self.use_bias = use_bias


class InterpNetForSequenceClassification(PreTrainedModel):
    config_class = InterpNetConfig
    all_tied_weights_keys = {}

    def __init__(self, config: InterpNetConfig):
        super().__init__(config)
        if config.n_layers > 0:
            self.input_proj = nn.Linear(config.input_size, config.hidden_size, bias=config.use_bias)
            self.transformer = Transformer(
                hidden_size=config.hidden_size,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                expansion_ratio=config.expansion_ratio,
                dropout=config.dropout,
                rotary=config.rotary,
                use_bias=config.use_bias,
                attention_backend=config.attention_backend,
            )

        self.get_logits = AttentionLogitsSequence(
            hidden_size=config.hidden_size if config.n_layers > 0 else config.input_size,
            num_labels=config.num_labels,
            sim_type=config.sim_type,
        )
        self.n_layers = config.n_layers
        self.num_labels = config.num_labels
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)
        self.config = config

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: Optional[bool] = None,
    ) -> InterpNetOutput:
        transformer_hidden_states = None
        transformer_s_max = None

        if self.n_layers > 0:
            embeddings = embeddings.to(next(self.input_proj.parameters()).dtype)
            x = self.input_proj(embeddings)
            if output_s_max is None:
                output_s_max = self.config.output_s_max
            transformer_outputs = self.transformer(
                hidden_states=x,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_s_max=output_s_max,
            )
            x = transformer_outputs.last_hidden_state
            transformer_hidden_states = transformer_outputs.hidden_states
            transformer_s_max = transformer_outputs.s_max
        else:
            embeddings = embeddings.to(next(self.get_logits.parameters()).dtype)
            x = embeddings
            if output_s_max is None:
                output_s_max = self.config.output_s_max

        logits, sims, x = self.get_logits(x, attention_mask)
        hidden_state_output = transformer_hidden_states if self.n_layers > 0 else (x,)
        if self.task_type == "sigmoid_regression":
            logits = logits.sigmoid()

        loss = None
        if labels is not None:
            if self.task_type == "regression":
                loss = self.loss_fct(logits.flatten(), labels.view(-1).float())
            elif self.task_type == "sigmoid_regression":
                loss = self.loss_fct(logits.flatten(), labels.view(-1).float())
            elif self.task_type == "multilabel":
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())

        return InterpNetOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_state_output if output_hidden_states else None,
            attentions=sims if output_attentions else None,
            s_max=transformer_s_max if output_s_max else None,
        )


class InterpNetForTokenClassification(PreTrainedModel):
    config_class = InterpNetConfig
    all_tied_weights_keys = {}

    def __init__(self, config: InterpNetConfig):
        super().__init__(config)
        if config.n_layers > 0:
            self.input_proj = nn.Linear(config.input_size, config.hidden_size, bias=config.use_bias)
            self.transformer = Transformer(
                hidden_size=config.hidden_size,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                expansion_ratio=config.expansion_ratio,
                dropout=config.dropout,
                rotary=config.rotary,
                use_bias=config.use_bias,
                attention_backend=config.attention_backend,
            )

        self.get_logits = AttentionLogitsToken(
            hidden_size=config.hidden_size if config.n_layers > 0 else config.input_size,
            num_labels=config.num_labels,
            sim_type=config.sim_type,
        )
        self.n_layers = config.n_layers
        self.num_labels = config.num_labels
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)
        self.config = config

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_s_max: Optional[bool] = None,
        **kwargs,
    ) -> InterpNetOutput:
        del kwargs
        transformer_hidden_states = None
        transformer_attentions = None
        transformer_s_max = None

        if self.n_layers > 0:
            embeddings = embeddings.to(next(self.input_proj.parameters()).dtype)
            x = self.input_proj(embeddings)
            if output_s_max is None:
                output_s_max = self.config.output_s_max
            transformer_outputs = self.transformer(
                hidden_states=x,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_s_max=output_s_max,
            )
            x = transformer_outputs.last_hidden_state
            transformer_hidden_states = transformer_outputs.hidden_states
            transformer_attentions = transformer_outputs.attentions
            transformer_s_max = transformer_outputs.s_max
        else:
            embeddings = embeddings.to(next(self.get_logits.parameters()).dtype)
            x = embeddings
            if output_s_max is None:
                output_s_max = self.config.output_s_max

        logits = self.get_logits(x, attention_mask)
        if self.task_type == "sigmoid_regression":
            logits = logits.sigmoid()

        loss = None
        if labels is not None:
            if self.task_type == "regression":
                loss = self.loss_fct(logits.flatten(), labels.view(-1).float())
            elif self.task_type == "sigmoid_regression":
                loss = self.loss_fct(logits.flatten(), labels.view(-1).float())
            elif self.task_type == "multilabel":
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())

        return InterpNetOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_hidden_states if output_hidden_states else None,
            attentions=transformer_attentions if output_attentions else None,
            s_max=transformer_s_max if output_s_max else None,
        )
