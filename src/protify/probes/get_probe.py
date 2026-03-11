from dataclasses import dataclass, field
from typing import List
from .linear_probe import LinearProbe, LinearProbeConfig
from .transformer_probe import TransformerForSequenceClassification, TransformerForTokenClassification, TransformerProbeConfig
from .interpnet import InterpNetForSequenceClassification, InterpNetForTokenClassification, InterpNetConfig
from .lyra_probe import LyraForSequenceClassification, LyraForTokenClassification, LyraConfig


@dataclass
class ProbeArguments:
    def __init__(
            self,
            probe_type: str = 'linear', # valid options: linear, transformer, interpnet
            tokenwise: bool = False,
            ### Linear Probe
            input_size: int = 960,
            hidden_size: int = 8192,
            dropout: float = 0.2,
            num_labels: int = 2,
            n_layers: int = 1,
            task_type: str = 'singlelabel',
            pre_ln: bool = True,
            sim_type: str = 'dot',
            use_bias: bool = False,
            add_token_ids: bool = False,
            ### Transformer Probe
            classifier_size: int = 4096,
            transformer_dropout: float = 0.1,
            classifier_dropout: float = 0.2,
            n_heads: int = 4,
            rotary: bool = True,
            attention_backend: str = "flex",
            output_s_max: bool = False,
            probe_pooling_types: List[str] = field(default_factory=lambda: ['mean', 'cls']),
            ### InterpNet
            # TODO
            ### LoRA
            lora: bool = False,
            lora_r: int = 8,
            lora_alpha: float = 32.0,
            lora_dropout: float = 0.01,
            **kwargs,

    ):
        self.probe_type = probe_type
        self.tokenwise = tokenwise
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.sim_type = sim_type
        self.add_token_ids = add_token_ids
        self.task_type = task_type
        self.pre_ln = pre_ln
        self.use_bias = use_bias
        self.classifier_size = classifier_size
        self.transformer_dropout = transformer_dropout
        self.classifier_dropout = classifier_dropout
        self.n_heads = n_heads
        self.rotary = rotary
        self.attention_backend = attention_backend
        self.output_s_max = output_s_max
        self.pooling_types = probe_pooling_types
        self.lora = lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout


def get_probe(args: ProbeArguments):
    if args.probe_type == 'linear' and not args.tokenwise:
        config = LinearProbeConfig(**args.__dict__)
        return LinearProbe(config)
    elif args.probe_type == 'transformer' and not args.tokenwise:
        config = TransformerProbeConfig(**args.__dict__)
        return TransformerForSequenceClassification(config)
    elif args.probe_type == 'transformer' and args.tokenwise:
        config = TransformerProbeConfig(**args.__dict__)
        return TransformerForTokenClassification(config)
    elif args.probe_type == 'interpnet' and not args.tokenwise:
        config = InterpNetConfig(**args.__dict__)
        return InterpNetForSequenceClassification(config)
    elif args.probe_type == 'interpnet' and args.tokenwise:
        config = InterpNetConfig(**args.__dict__)
        return InterpNetForTokenClassification(config)
    elif args.probe_type == 'lyra' and not args.tokenwise:
        config = LyraConfig(**args.__dict__)
        return LyraForSequenceClassification(config)
    elif args.probe_type == 'lyra' and args.tokenwise:
        config = LyraConfig(**args.__dict__)
        return LyraForTokenClassification(config)
    else:
        raise ValueError(f"Invalid combination of probe type and tokenwise: {args.probe_type} {args.tokenwise}")


def rebuild_probe_from_saved_config(
        probe_type: str,
        tokenwise: bool,
        probe_config: dict,
    ):
    config_dict = dict(probe_config)
    if "num_labels" not in config_dict and "id2label" in config_dict:
        config_dict["num_labels"] = len(config_dict["id2label"])
    if "pooling_types" in config_dict and "probe_pooling_types" not in config_dict:
        config_dict["probe_pooling_types"] = config_dict["pooling_types"]

    if probe_type == "linear" and not tokenwise:
        config = LinearProbeConfig(**config_dict)
        return LinearProbe(config)
    if probe_type == "transformer" and not tokenwise:
        config = TransformerProbeConfig(**config_dict)
        return TransformerForSequenceClassification(config)
    if probe_type == "transformer" and tokenwise:
        config = TransformerProbeConfig(**config_dict)
        return TransformerForTokenClassification(config)
    if probe_type == "interpnet" and not tokenwise:
        config = InterpNetConfig(**config_dict)
        return InterpNetForSequenceClassification(config)
    if probe_type == "interpnet" and tokenwise:
        config = InterpNetConfig(**config_dict)
        return InterpNetForTokenClassification(config)
    if probe_type == "lyra" and not tokenwise:
        config = LyraConfig(**config_dict)
        return LyraForSequenceClassification(config)
    if probe_type == "lyra" and tokenwise:
        config = LyraConfig(**config_dict)
        return LyraForTokenClassification(config)
    raise ValueError(f"Unsupported saved probe configuration: {probe_type} tokenwise={tokenwise}")
