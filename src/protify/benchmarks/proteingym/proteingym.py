'''
from .data_loader import ProteinGymCVSplitter, load_proteingym_dms
from typing import List, Dict
import pandas as pd
from dataclasses import dataclass
from ...utils import print_message
from ...embedder import Embedder, EmbeddingArguments
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr, pearsonr
import torch



@dataclass
class ProteinGymArgs:
    repo_id: str = 'nikraf/ProteinGym_DMS'
    n_folds: int = 5
    seed: int = 42
    task_type: str = 'regression'
    plots_dir: str = 'results/proteingym'
    patience: int = 5


class ProteinGymBenchmark:
    def __init__(self, args: ProteinGymArgs):
        self.args = args
        self.zero_shot_evaluator = ProteinGymZeroShot()
        self.cv_splitter = ProteinGymCVSplitter()

    def _build_fold_datasets(self, df: pd.DataFrame, idxs: Dict[str, list]) -> tuple:
        # Convert to HF-style minimal datasets per fold with columns 'seqs' and 'labels'
        def make_dataset(sub: pd.DataFrame):
            return {
                'seqs': sub['mutant_sequence'].tolist(),
                'labels': sub['labels'].tolist(),
            }
        train_df = df.iloc[idxs['train_idx']]
        valid_df = df.iloc[idxs['valid_idx']]
        test_df = df.iloc[idxs['test_idx']]
        return make_dataset(train_df), make_dataset(valid_df), make_dataset(test_df)

    def run_benchmark(self, model_names: List[str], dms_ids: List[str]) -> pd.DataFrame:
        """
        1) Load ProteinGym tables per DMS
        2) Zero-shot delta log-prob evaluation
        3) Supervised 5-fold CV with 3 strategies
        Return tidy results DataFrame
        """
        records = []
        for dms_id in dms_ids:
            print_message(f'Loading ProteinGym assay: {dms_id}')
            # NOTE: repo_id optional, loader is pinned to HF dataset
            df = load_proteingym_dms(dms_id, repo_id=self.args.repo_id)
            # Ensure mutant sequences present
            df = self.zero_shot_evaluator.ensure_mutant_sequences(df)

            # Zero-shot for each model
            for model_name in model_names:
                zs = self.zero_shot_evaluator.score_assay(df, model_name)
                rho = zs['delta_log_prob'].corr(df['labels'], method='spearman')
                r = zs['delta_log_prob'].corr(df['labels'], method='pearson')
                records.append({
                    'DMS_id': dms_id,
                    'model': model_name,
                    'mode': 'zero-shot',
                    'metric': 'spearman',
                    'value': rho,
                })
                records.append({
                    'DMS_id': dms_id,
                    'model': model_name,
                    'mode': 'zero-shot',
                    'metric': 'pearson',
                    'value': r,
                })

            # Supervised CV per strategy using mean-pooled embeddings + Ridge
            n = len(df)
            strategies = {
                'random': self.cv_splitter.random_split(n, self.args.n_folds, self.args.seed),
                'contiguous': self.cv_splitter.contiguous_split(n, self.args.n_folds),
                'modulo': self.cv_splitter.modulo_split(n, self.args.n_folds),
            }
            for model_name in model_names:
                # Embed all mutant sequences once per model
                all_mutants = df['mutant_sequence'].tolist()
                embedder_args = EmbeddingArguments(
                    embedding_batch_size=8,
                    embedding_num_workers=0,
                    download_embeddings=False,
                    matrix_embed=False,
                    embedding_pooling_types=['mean'],
                    save_embeddings=False,
                    embed_dtype=torch.float32,
                    sql=False,
                    embedding_save_dir='embeddings',
                )
                embedder = Embedder(embedder_args, all_mutants)
                emb_dict = embedder(model_name) or {}
                # Ensure we have in-memory embeddings
                if emb_dict is None or len(emb_dict) == 0:
                    raise RuntimeError(f"No embeddings produced for model {model_name}")

                # Build matrix X once
                X = np.stack([emb_dict[seq].cpu().numpy().squeeze(0) for seq in all_mutants], axis=0)
                y = df['labels'].to_numpy(dtype=float)

                for strategy_name, splits in strategies.items():
                    for fold_idx, split in enumerate(splits):
                        tr, va, te = split['train_idx'], split['valid_idx'], split['test_idx']
                        # Train on train+valid, report on test per ProteinGym convention
                        train_idx = np.concatenate([tr, va])
                        X_tr, y_tr = X[train_idx], y[train_idx]
                        X_te, y_te = X[te], y[te]
                        model = Ridge(alpha=1.0)
                        model.fit(X_tr, y_tr)
                        y_pred = model.predict(X_te)
                        rho = spearmanr(y_te, y_pred).correlation
                        pr = pearsonr(y_te, y_pred)
                        r = pr.statistic if hasattr(pr, 'statistic') else pr[0]
                        records.append({
                            'DMS_id': dms_id,
                            'model': model_name,
                            'mode': f'cv-{strategy_name}',
                            'fold': fold_idx,
                            'metric': 'spearman',
                            'value': rho,
                        })
                        records.append({
                            'DMS_id': dms_id,
                            'model': model_name,
                            'mode': f'cv-{strategy_name}',
                            'fold': fold_idx,
                            'metric': 'pearson',
                            'value': r,
                        })

        return pd.DataFrame.from_records(records)

'''