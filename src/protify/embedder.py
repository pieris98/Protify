"""Embedding utilities for protein sequences using pre-trained models."""

import os
import gzip
import sqlite3
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from seed_utils import seed_worker, dataloader_generator, get_global_seed
    from data.dataset_classes import SimpleProteinDataset
    from base_models.get_base_models import get_base_model
    from pooler import Pooler
    from utils import torch_load, print_message
except ImportError:
    from .seed_utils import seed_worker, dataloader_generator, get_global_seed
    from .data.dataset_classes import SimpleProteinDataset
    from .base_models.get_base_models import get_base_model
    from .pooler import Pooler
    from .utils import torch_load, print_message


def build_collator(tokenizer) -> Callable[[List[str]], dict[str, torch.Tensor]]:
    """Create a collate function for batching protein sequences."""
    def _collate_fn(sequences: List[str]) -> dict[str, torch.Tensor]:
        return tokenizer(sequences, return_tensors="pt", padding='longest', pad_to_multiple_of=8)
    return _collate_fn


def get_embedding_filename(model_name: str, matrix_embed: bool, pooling_types: List[str], ext: str = 'pth') -> str:
    """
    Generate embedding filename.
    
    Format: {model_name}_{matrix_embed}[_{pooling}].{ext}
    Pooling types are only included for vector (non-matrix) embeddings.
    """
    base = f'{model_name}_{matrix_embed}'
    if not matrix_embed and pooling_types:
        base = f'{base}_{"_".join(sorted(pooling_types))}'
    return f'{base}.{ext}'


@dataclass
class EmbeddingArguments:
    """Configuration for the Embedder."""
    embedding_batch_size: int = 4
    embedding_num_workers: int = 0
    download_embeddings: bool = False
    download_dir: str = 'Synthyra/vector_embeddings'
    matrix_embed: bool = False
    embedding_pooling_types: List[str] = field(default_factory=lambda: ['mean'])
    save_embeddings: bool = False
    embed_dtype: torch.dtype = torch.float32
    sql: bool = False
    embedding_save_dir: str = 'embeddings'

    # Aliases for cleaner internal access
    @property
    def batch_size(self) -> int:
        return self.embedding_batch_size

    @property
    def num_workers(self) -> int:
        return self.embedding_num_workers

    @property
    def pooling_types(self) -> List[str]:
        return self.embedding_pooling_types


class Embedder:
    """
    Embed protein sequences using pre-trained language models.
    
    Supports caching to disk (pth or SQLite), downloading pre-computed 
    embeddings, and various pooling strategies for vector embeddings.
    """

    def __init__(self, args: EmbeddingArguments, all_seqs: List[str], **kwargs):
        self.args = args
        self.all_seqs = all_seqs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print_message(f'Device {self.device} found')

    def _get_save_path(self, model_name: str) -> str:
        """Get the save path for embeddings."""
        ext = 'db' if self.args.sql else 'pth'
        filename = get_embedding_filename(model_name, self.args.matrix_embed, self.args.pooling_types, ext)
        return os.path.join(self.args.embedding_save_dir, filename)

    def _download_embeddings(self, model_name: str) -> Optional[str]:
        """Download and merge pre-computed embeddings from HuggingFace Hub."""
        filename = get_embedding_filename(model_name, self.args.matrix_embed, self.args.pooling_types, 'pth')
        
        try:
            local_path = hf_hub_download(
                repo_id=self.args.download_dir,
                filename=f'embeddings/{filename}.gz',
                repo_type='dataset'
            )
        except Exception:
            print_message(f'No embeddings found for {model_name} in {self.args.download_dir}')
            return None

        # Decompress
        print_message(f'Unzipping {local_path}')
        unzipped_path = local_path.replace('.gz', '')
        with gzip.open(local_path, 'rb') as f_in:
            with open(unzipped_path, 'wb') as f_out:
                f_out.write(f_in.read())

        final_path = os.path.join(self.args.embedding_save_dir, filename)
        downloaded = torch_load(unzipped_path)

        # Merge with existing embeddings if present
        if os.path.exists(final_path):
            print_message(f'Merging with existing embeddings in {final_path}')
            existing = torch_load(final_path)
            
            if self.args.embed_dtype != torch.float16:
                print_message(
                    f"Warning: Downloaded embeddings are float16 but current dtype is {self.args.embed_dtype}. "
                    "This could affect performance."
                )
            
            downloaded.update(existing)
            for seq in downloaded:
                downloaded[seq] = downloaded[seq].to(self.args.embed_dtype)
        
        print_message(f'Saving embeddings to {final_path}')
        torch.save(downloaded, final_path)
        return final_path

    def _load_existing_embeddings(self, model_name: str) -> tuple[List[str], str, dict]:
        """
        Load existing embeddings and determine which sequences need embedding.
        
        Returns:
            Tuple of (sequences_to_embed, save_path, existing_embeddings_dict)
        """
        save_path = self._get_save_path(model_name)

        if self.args.sql:
            return self._load_sql_embeddings(save_path)
        return self._load_pth_embeddings(save_path)

    def _load_sql_embeddings(self, save_path: str) -> tuple[List[str], str, dict]:
        """Load embeddings from SQLite database."""
        if not os.path.exists(save_path):
            print_message(f"No embeddings found in {save_path}")
            return self.all_seqs, save_path, {}

        with sqlite3.connect(save_path) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)')
            cursor = conn.execute("SELECT sequence FROM embeddings")
            existing = {row[0] for row in cursor}

        to_embed = [seq for seq in self.all_seqs if seq not in existing]
        print_message(f"Loaded {len(existing)} embeddings from {save_path}\nEmbedding {len(to_embed)} new sequences")
        return to_embed, save_path, {}

    def _load_pth_embeddings(self, save_path: str) -> tuple[List[str], str, dict]:
        """Load embeddings from PyTorch file."""
        if not os.path.exists(save_path):
            print_message(f"No embeddings found in {save_path}")
            return self.all_seqs, save_path, {}

        print_message(f"Loading embeddings from {save_path}")
        embeddings = torch_load(save_path)
        print_message(f"Loaded {len(embeddings)} embeddings")
        
        to_embed = [seq for seq in self.all_seqs if seq not in embeddings]
        return to_embed, save_path, embeddings

    def _get_attention_mask(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Extract or create attention mask from batch."""
        if 'attention_mask' in batch:
            return batch['attention_mask']
        if 'sequence_ids' in batch:
            return (batch['sequence_ids'] != -1).long().to(device)
        return torch.ones_like(batch['input_ids'], device=device)

    @torch.inference_mode()
    def _embed_sequences(
        self,
        to_embed: List[str],
        save_path: str,
        model: torch.nn.Module,
        tokenizer,
        embeddings_dict: dict[str, torch.Tensor]
    ) -> Optional[dict[str, torch.Tensor]]:
        """Embed sequences and optionally save to disk."""
        os.makedirs(self.args.embedding_save_dir, exist_ok=True)
        
        model = model.to(self.device).eval()
        if os.name == 'posix':
            try:
                torch.compile(model)
            except Exception:
                print_message("Model cannot be compiled")

        print_message(f'Pooling types: {self.args.pooling_types}')
        pooler = None if self.args.matrix_embed else Pooler(self.args.pooling_types)
        collate_fn = build_collator(tokenizer)
        
        dataset = SimpleProteinDataset(to_embed)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            prefetch_factor=2 if self.args.num_workers > 0 else None,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=dataloader_generator(get_global_seed())
        )

        # Initialize SQL connection if needed
        conn = None
        if self.args.sql:
            conn = sqlite3.connect(save_path)
            conn.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)')

        use_parti = 'parti' in self.args.pooling_types

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding'):
            seqs = to_embed[i * self.args.batch_size : (i + 1) * self.args.batch_size]
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            attention_mask = self._get_attention_mask(batch, self.device)

            # Get embeddings from model
            if use_parti:
                try:
                    residue_emb, attentions = model(**batch, output_attentions=True)
                    embeddings = self._pool(residue_emb, attention_mask, pooler, attentions)
                except Exception as e:
                    print_message(f"Parti pooling failed: {e}\nFalling back to mean pooling")
                    pooler = Pooler(['mean'])
                    use_parti = False
                    residue_emb = model(**batch)
                    embeddings = self._pool(residue_emb, attention_mask, pooler)
            else:
                residue_emb = model(**batch)
                embeddings = self._pool(residue_emb, attention_mask, pooler)

            # Store embeddings
            for seq, emb, mask in zip(seqs, embeddings, attention_mask.cpu()):
                emb = emb[mask.bool()] if self.args.matrix_embed else emb
                
                if self.args.sql:
                    conn.execute(
                        "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                        (seq, emb.numpy().tobytes())
                    )
                else:
                    embeddings_dict[seq] = emb.to(self.args.embed_dtype)

            if self.args.sql and (i + 1) % 100 == 0:
                conn.commit()

        if self.args.sql:
            conn.commit()
            conn.close()
            return None

        if self.args.save_embeddings:
            print_message(f"Saving embeddings to {save_path}")
            torch.save(embeddings_dict, save_path)

        return embeddings_dict

    def _pool(
        self,
        residue_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        pooler: Optional[Pooler],
        attentions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply pooling to residue embeddings."""
        if residue_emb.ndim == 2 or self.args.matrix_embed:
            return residue_emb.cpu()
        return pooler(emb=residue_emb, attention_mask=attention_mask, attentions=attentions).cpu()

    def __call__(self, model_name: str) -> Optional[dict[str, torch.Tensor]]:
        """
        Embed sequences with the specified model.
        
        Returns dict mapping sequences to embeddings, or None if using SQL storage.
        """
        clean_name = model_name.split('---')[-1].split('/')[-1] if 'custom' in model_name.lower() else model_name

        if self.args.download_embeddings:
            self._download_embeddings(clean_name)

        if self.device == torch.device('cpu'):
            warnings.warn("Embedding on CPU is slow. Consider downloading pre-computed embeddings.")

        to_embed, save_path, embeddings_dict = self._load_existing_embeddings(clean_name)

        if not to_embed:
            print_message(f"All sequences already embedded for {clean_name}")
            return embeddings_dict

        print_message(f"Embedding {len(to_embed)} sequences with {clean_name}")
        model, tokenizer = get_base_model(model_name)
        return self._embed_sequences(to_embed, save_path, model, tokenizer, embeddings_dict)


if __name__ == '__main__':
    # Embed supported datasets with supported models
    # Usage: py -m embedder
    import argparse
    from huggingface_hub import upload_file, login
    from data.supported_datasets import vector_benchmark
    from data.data_mixin import DataArguments, DataMixin
    from base_models.get_base_models import BaseModelArguments
    from seed_utils import set_global_seed

    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    parser = argparse.ArgumentParser(description='Embed protein sequences and upload to HuggingFace Hub')
    parser.add_argument('--token', default=None, help='HuggingFace token')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--embed_dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--model_names', nargs='+', default=['standard'])
    parser.add_argument('--models_to_skip', nargs='+', default=[], help='Skip downloading for these models')
    parser.add_argument('--embedding_save_dir', type=str, default='embeddings')
    parser.add_argument('--download_dir', type=str, default='Synthyra/vector_embeddings')
    parser.add_argument('--embedding_pooling_types', nargs='+', default=['mean', 'var'])
    args = parser.parse_args()

    set_global_seed()

    if args.token:
        login(args.token)

    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    dtype = dtype_map[args.embed_dtype]

    # Load sequences
    data_args = DataArguments(data_names=vector_benchmark, max_length=1024, trim=False)
    all_seqs = DataMixin(data_args).get_data()[1]

    # Embed with each model
    model_args = BaseModelArguments(model_names=args.model_names)
    for model_name in model_args.model_names:
        embedder_args = EmbeddingArguments(
            embedding_batch_size=args.batch_size,
            embedding_num_workers=args.num_workers,
            download_embeddings=model_name not in args.models_to_skip,
            matrix_embed=False,
            embedding_pooling_types=args.embedding_pooling_types,
            save_embeddings=True,
            embed_dtype=dtype,
            sql=False,
            embedding_save_dir=args.embedding_save_dir
        )
        embedder = Embedder(embedder_args, all_seqs)
        _ = embedder(model_name)

        # Compress and upload
        filename = get_embedding_filename(model_name, False, embedder_args.pooling_types, 'pth')
        save_path = os.path.join(args.embedding_save_dir, filename)
        compressed_path = f"{save_path}.gz"
        
        print(f"Compressing {save_path}")
        with open(save_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.write(f_in.read())

        upload_file(
            path_or_fileobj=compressed_path,
            path_in_repo=f'embeddings/{filename}.gz',
            repo_id=args.download_dir,
            repo_type='dataset'
        )

    print('Done')
