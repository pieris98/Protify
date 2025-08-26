"""
Utility functions for hyperparameter optimization in Protify.
"""

import copy
import json
import wandb
from typing import Dict, Any, List, Tuple


def select_metric(metrics: Dict[str, Any], sweep_metric: str) -> float:
    """
    Function to select the metric to optimize
    """
    if metrics is None:
        raise ValueError("Metrics are None")
    
    if sweep_metric in metrics and metrics[sweep_metric] is not None:
        try:
            return float(metrics[sweep_metric])
        except Exception as e:
            raise ValueError(f"Error converting {sweep_metric} to float: {e}")

def apply_config(cfg: Dict[str, Any], probe_args: Any, trainer_args: Any, 
                probe_keys: set, trainer_keys: set) -> None:
    """
    Function to apply sweep config values to probe and trainer arguments
    """
    for k, v in cfg.items():
        if k in probe_keys and hasattr(probe_args, k):
            setattr(probe_args, k, v)
        if k in trainer_keys and hasattr(trainer_args, k):
            setattr(trainer_args, k, v)


def create_objective_function(model_name: str, data_name: str, dataset: Tuple, 
                            base_probe: Dict[str, Any], base_trainer: Dict[str, Any],
                            probe_args: Any, trainer_args: Any, full_args: Any,
                            sweep_config: Dict[str, Any], probe_keys: set, trainer_keys: set,
                            emb_dict: Any, ppi: bool, random_id: str, results_list: List[Dict[str, Any]],
                            get_base_model_for_training, get_probe, wrap_lora,
                            trainer_base_model, trainer_hybrid_model, trainer_probe):
    """
    Create and return the objective function for hyperparameter optimization.
    
    Args:
        model_name: Name of the model
        data_name: Name of the dataset
        dataset: Tuple containing train, valid, test sets and metadata
        base_probe: Base probe arguments
        base_trainer: Base trainer arguments
        probe_args: Probe arguments object
        trainer_args: Trainer arguments object
        full_args: Full arguments object
        sweep_config: Sweep configuration
        probe_keys: Set of valid probe configuration keys
        trainer_keys: Set of valid trainer configuration keys
        emb_dict: Embedding dictionary
        ppi: Whether this is a PPI task
        results_list: List to store results
        random_id: Random identifier for logging
        get_base_model_for_training: Function to get base model for training
        get_probe: Function to get probe
        wrap_lora: Function to wrap model with LoRA
        trainer_base_model: Function to train base model
        trainer_hybrid_model: Function to train hybrid model
        trainer_probe: Function to train probe
        
    Returns:
        function: The objective function for hyperparameter optimization
    """
    train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
    
    def objective():
        run = wandb.init(
            project=full_args.wandb_project,
            entity=full_args.wandb_entity,
            config=sweep_config,
            reinit=True,
            tags=["sweep", f"model:{model_name}", f"data:{data_name}"],
        )
        run.name = f"sweep-{model_name}_{data_name}-{run.id[:6]}"
        try:
            # Reset args, apply sweep cfg
            probe_args.__dict__.update(copy.deepcopy(base_probe))
            trainer_args.__dict__.update(copy.deepcopy(base_trainer))
            trainer_args.make_plots = False
            trainer_args.sweep_mode = True

            cfg = dict(wandb.config)
            apply_config(cfg, probe_args, trainer_args, probe_keys, trainer_keys)

            if full_args.full_finetuning:
                model, tokenizer = get_base_model_for_training(
                    model_name,
                    tokenwise=probe_args.tokenwise,
                    num_labels=probe_args.num_labels,
                    hybrid=False,
                )
                if probe_args.lora:
                    model = wrap_lora(model, probe_args.lora_r, probe_args.lora_alpha, probe_args.lora_dropout)
                _, valid_metrics, test_metrics = trainer_base_model(
                    model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    data_name=data_name,
                    train_dataset=train_set,
                    valid_dataset=valid_set,
                    test_dataset=test_set,
                    ppi=ppi,
                    log_id=random_id,
                )

            elif full_args.hybrid_probe:
                model, tokenizer = get_base_model_for_training(
                    model_name,
                    tokenwise=probe_args.tokenwise,
                    num_labels=probe_args.num_labels,
                    hybrid=True,
                )
                if probe_args.lora:
                    model = wrap_lora(model, probe_args.lora_r, probe_args.lora_alpha, probe_args.lora_dropout)
                probe = get_probe(probe_args)
                _, valid_metrics, test_metrics = trainer_hybrid_model(
                    model=model,
                    tokenizer=tokenizer,
                    probe=probe,
                    model_name=model_name,
                    data_name=data_name,
                    train_dataset=train_set,
                    valid_dataset=valid_set,
                    test_dataset=test_set,
                    emb_dict=emb_dict,
                    ppi=ppi,
                    log_id=random_id,
                )

            else:  # nn probe
                _, tokenizer = get_base_model_for_training(
                    model_name,
                    tokenwise=probe_args.tokenwise,
                    num_labels=probe_args.num_labels,
                    hybrid=False,
                )
                probe = get_probe(probe_args)
                _, valid_metrics, test_metrics = trainer_probe(
                    model=probe,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    data_name=data_name,
                    train_dataset=train_set,
                    valid_dataset=valid_set,
                    test_dataset=test_set,
                    emb_dict=emb_dict,
                    ppi=ppi,
                    log_id=random_id,
                )

            # Log & return sweep metric
            all_metrics = {}
            if isinstance(valid_metrics, dict):
                for k, v in valid_metrics.items():
                    all_metrics[f"{k}"] = v
            if isinstance(test_metrics, dict):
                for k, v in test_metrics.items():
                    all_metrics[f"{k}"] = v
            wandb.log(all_metrics)
            metric_value = select_metric(valid_metrics, full_args.sweep_metric)
            results_list.append({
                "wandb_run_id": run.id,
                full_args.sweep_metric: metric_value,
                "config": dict(run.config),
                "valid_metrics": valid_metrics,
                "test_metrics": test_metrics,
            })
            return float(metric_value)
        finally:
            run.finish()
    return objective
