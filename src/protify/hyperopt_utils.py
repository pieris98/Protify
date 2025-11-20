import copy
import wandb
import os
import yaml
import json
import csv
from typing import Dict, Any, List, Tuple
from utils import torch_load, print_message
from embedder import get_embedding_filename
from base_models.get_base_models import get_tokenizer

class HyperoptModule:
    def __init__(
        self, 
        main_process,
        model_name: str,
        data_name: str,
        dataset: Tuple,
        emb_dict: Any,
        sweep_config: Dict[str, Any],
        results_list: List[Dict[str, Any]]
    ):
        self.mp = main_process
        self.model_name = model_name
        self.data_name = data_name
        self.dataset = dataset
        self.emb_dict = emb_dict
        self.sweep_config = sweep_config
        self.results_list = results_list
        
        self.base_probe_args = copy.deepcopy(self.mp.probe_args.__dict__)
        self.base_trainer_args = copy.deepcopy(self.mp.trainer_args.__dict__)
        
        self.probe_keys = {
            'hidden_size','dropout','n_layers','pre_ln','classifier_dim',
            'transformer_dropout','classifier_dropout','n_heads','rotary',
            'lora','lora_r','lora_alpha','lora_dropout','probe_type','tokenwise'
        }
        self.trainer_keys = {
            'lr','weight_decay','num_epochs','probe_batch_size',
            'base_batch_size','probe_grad_accum','base_grad_accum',
            'patience','seed'
        }

    def apply_config(self, cfg: Dict[str, Any]):
        self.mp.probe_args.__dict__.update(copy.deepcopy(self.base_probe_args))
        self.mp.trainer_args.__dict__.update(copy.deepcopy(self.base_trainer_args))
        
        for k, v in cfg.items():
            if k in self.probe_keys and hasattr(self.mp.probe_args, k):
                setattr(self.mp.probe_args, k, v)
            if k in self.trainer_keys and hasattr(self.mp.trainer_args, k):
                setattr(self.mp.trainer_args, k, v)

    def train_model(self, sweep_mode=True):
        train_set, valid_set, test_set, _, _, ppi = self.dataset
        
        if self.mp.full_args.full_finetuning:
             model, valid_metrics, test_metrics = self.mp._run_full_finetuning(
                self.model_name, self.data_name, 
                train_set, valid_set, test_set, 
                ppi=ppi, sweep_mode=sweep_mode
            )
             return model, valid_metrics, test_metrics

        elif self.mp.full_args.hybrid_probe:
            tokenizer = get_tokenizer(self.model_name)
            model, valid_metrics, test_metrics = self.mp._run_hybrid_probe(
                self.model_name, self.data_name,
                train_set, valid_set, test_set,
                tokenizer,
                emb_dict=self.emb_dict,
                ppi=ppi,
                sweep_mode=sweep_mode
            )
            return model, valid_metrics, test_metrics

        else:
            tokenizer = get_tokenizer(self.model_name)
            probe, valid_metrics, test_metrics = self.mp._run_nn_probe(
                self.model_name, self.data_name,
                train_set, valid_set, test_set,
                tokenizer, 
                emb_dict=self.emb_dict, 
                ppi=ppi, 
                sweep_mode=sweep_mode
            )
            return probe, valid_metrics, test_metrics

    def select_metric(self, metrics: Dict[str, Any], sweep_metric: str) -> float:
        return float(metrics[sweep_metric])

    def objective(self):
        run = wandb.init(
            project=self.mp.full_args.wandb_project,
            entity=self.mp.full_args.wandb_entity,
            config=self.sweep_config,
            reinit=True,
            tags=["sweep", f"model:{self.model_name}", f"data:{self.data_name}"],
        )
        run.name = f"sweep-{self.model_name}_{self.data_name}-{run.id[:6]}"
        
        self.apply_config(dict(wandb.config))
        self.mp.trainer_args.make_plots = False
        self.mp.trainer_args.sweep_mode = True
        
        _, valid_metrics, test_metrics = self.train_model(sweep_mode=True)
        
        # Choose task-specific metric to optimize
        label_type = self.mp.probe_args.task_type
        metric_cls = getattr(self.mp.full_args, 'sweep_metric_cls', None)
        metric_reg = getattr(self.mp.full_args, 'sweep_metric_reg', None)
        dataset_metric = metric_cls if label_type in ["singlelabel", "multilabel"] else metric_reg

        all_metrics = {}
        if isinstance(valid_metrics, dict):
            for k, v in valid_metrics.items():
                all_metrics[f"{k}"] = v
        if isinstance(test_metrics, dict):
            for k, v in test_metrics.items():
                all_metrics[f"{k}"] = v
        wandb.log(all_metrics)
        
        metric_value = self.select_metric(valid_metrics, dataset_metric)
        
        self.results_list.append({
            "wandb_run_id": run.id,
            dataset_metric: metric_value,
            "config": dict(run.config),
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        })
        
        run.finish()
        return float(metric_value)

    @classmethod
    def run_wandb_hyperopt(cls, mp):
        mp.logger.info("Called method: run_wandb_hyperopt")

        sweep_config = {}
        sweep_config_path = mp.full_args.sweep_config_path
            
        if os.path.exists(sweep_config_path):
            with open(sweep_config_path, 'r') as f:
                sweep_config = yaml.safe_load(f)
        else:
            raise ValueError(f"Sweep config file not found: {sweep_config_path}")

        params_to_hyperopt = sweep_config.get("parameters", {})

        method = mp.full_args.sweep_method
        early_term = sweep_config.get("early_terminate", None)

        total_combinations = len(mp.model_args.model_names) * len(mp.datasets)
        mp.logger.info(f"Hyperopt over {total_combinations} model/dataset combinations")
        for model_name in mp.model_args.model_names:
            tokenizer = get_tokenizer(model_name)
            test_seq = mp.all_seqs[0]

            if "random" in model_name.lower():
                print_message(f"Skipping hyperparameter optimization for {model_name}.")

                for data_name, dataset in mp.datasets.items():
                    train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
                    mp.probe_args.num_labels = num_labels
                    mp.probe_args.task_type = label_type
                    mp.trainer_args.task_type = label_type
                    mp.trainer_args.make_plots = True

                    emb_dict = None
                    if not mp.full_args.full_finetuning:
                        if mp._sql:
                            filename = get_embedding_filename(model_name, mp._full, mp.embedding_args.pooling_types, 'db')
                            save_path = os.path.join(mp.embedding_args.embedding_save_dir, filename)
                            input_dim = mp.get_embedding_dim_sql(save_path, test_seq, tokenizer)
                        else:
                            filename = get_embedding_filename(model_name, mp._full, mp.embedding_args.pooling_types, 'pth')
                            save_path = os.path.join(mp.embedding_args.embedding_save_dir, filename)
                            emb_dict = torch_load(save_path)
                            input_dim = mp.get_embedding_dim_pth(emb_dict, test_seq, tokenizer)
                        mp.probe_args.input_dim = input_dim * 2 if (ppi and not mp._full) else input_dim
                    if mp.full_args.full_finetuning:
                        _ = mp._run_full_finetuning(model_name, data_name, train_set, valid_set, test_set, ppi, sweep_mode=False)
                    elif mp.full_args.hybrid_probe:
                        _ = mp._run_hybrid_probe(model_name, data_name, train_set, valid_set, test_set, tokenizer, emb_dict=emb_dict, ppi=ppi, sweep_mode=False)
                    else:
                        _ = mp._run_nn_probe(model_name, data_name, train_set, valid_set, test_set, tokenizer, emb_dict=emb_dict, ppi=ppi, sweep_mode=False)
                continue

            for data_name, dataset in mp.datasets.items():
                mp.logger.info(f"Sweeping over {data_name} with {model_name}")
                train_set, _, _, num_labels, label_type, ppi = dataset
                mp.probe_args.num_labels = num_labels
                mp.probe_args.task_type = label_type
                mp.trainer_args.task_type = label_type

                emb_dict = None
                if not mp.full_args.full_finetuning:
                    if mp._sql:
                        filename = get_embedding_filename(model_name, mp._full, mp.embedding_args.pooling_types, 'db')
                        save_path = os.path.join(mp.embedding_args.embedding_save_dir, filename)
                        input_dim = mp.get_embedding_dim_sql(save_path, test_seq, tokenizer)
                    else:
                        filename = get_embedding_filename(model_name, mp._full, mp.embedding_args.pooling_types, 'pth')
                        save_path = os.path.join(mp.embedding_args.embedding_save_dir, filename)
                        emb_dict = torch_load(save_path)
                        input_dim = mp.get_embedding_dim_pth(emb_dict, test_seq, tokenizer)
                    mp.probe_args.input_dim = input_dim * 2 if (ppi and not mp._full) else input_dim

                # Save base args for restoring after each trial
                base_probe = copy.deepcopy(mp.probe_args.__dict__)
                base_trainer = copy.deepcopy(mp.trainer_args.__dict__)

                results_list = []
                # Choose task-specific metric to optimize
                metric_cls = getattr(mp.full_args, 'sweep_metric_cls', None)
                metric_reg = getattr(mp.full_args, 'sweep_metric_reg', None)
                dataset_metric = metric_cls if label_type in ["singlelabel", "multilabel"] else metric_reg
                
                hyperopt_module = cls(
                    main_process=mp,
                    model_name=model_name,
                    data_name=data_name,
                    dataset=dataset,
                    emb_dict=emb_dict,
                    sweep_config=sweep_config,
                    results_list=results_list
                )

                wb_sweep = {
                    "method": method,
                    "metric": {"name": dataset_metric, "goal": mp.full_args.sweep_goal},
                    "early_terminate": early_term,
                    "parameters": params_to_hyperopt,
                }
                sweep_id = wandb.sweep(sweep=wb_sweep, project=mp.full_args.wandb_project, entity=mp.full_args.wandb_entity)
                wandb.agent(sweep_id, function=hyperopt_module.objective, count=mp.full_args.sweep_count)

                # Sort, write, and save sweep results
                reverse_flag = True if mp.full_args.sweep_goal == "maximize" else False
                results_list.sort(key=lambda x: x[dataset_metric], reverse=reverse_flag)
                sweep_log_path = os.path.join(mp.full_args.log_dir, f"{mp.random_id}_sweep_{data_name}_{model_name}.csv")
                with open(sweep_log_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=',')
                    # Columns
                    columns = ["rank","wandb_run_id",dataset_metric,"config","valid_metrics","test_metrics"]
                    writer.writerow(columns)
                    for idx, res in enumerate(results_list, start=1):
                        writer.writerow([
                            idx,
                            res['wandb_run_id'],
                            res[dataset_metric],
                            json.dumps(res['config']),
                            json.dumps(res['valid_metrics']),
                            json.dumps(res['test_metrics']),
                        ])

                # Log best hyperparameters
                best = results_list[0] if results_list else None
                best_score = best[dataset_metric]
                best_config = best['config']
                print_message(f"Best sweep result - {dataset_metric}: {best_score}")
                print_message(f"Best hyperparameters: {json.dumps(best_config, indent=2)}")

                # Restore base args then apply best
                mp.probe_args.__dict__.update(copy.deepcopy(base_probe))
                mp.trainer_args.__dict__.update(copy.deepcopy(base_trainer))
                hyperopt_module.apply_config(best_config)
                mp.trainer_args.make_plots = True

                # Run best model with the best hyperparameters, log metrics, create plots
                hyperopt_module.train_model(sweep_mode=False)