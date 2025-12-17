"""Top-level package for Protify.

Exposes common subpackages for convenient imports whether Protify is used
as a submodule (installed or added to PYTHONPATH) or executed from source.
"""

# Re-export subpackages
from . import probes  # noqa: F401
from . import data  # noqa: F401
from . import base_models  # noqa: F401

# Re-export commonly used classes and functions
from .pooler import Pooler  # noqa: F401
from .utils import torch_load, print_message  # noqa: F401
from .seed_utils import (  # noqa: F401
    set_global_seed,
    get_global_seed,
    seed_worker,
    dataloader_generator,
    set_determinism,
)
from .data.data_mixin import DataArguments, DataMixin  # noqa: F401
from .embedder import Embedder, EmbeddingArguments  # noqa: F401

__all__ = [
    # Subpackages
    "probes",
    "data",
    "base_models",
    # Classes
    "Embedder",
    "EmbeddingArguments",
    "Pooler",
    "DataArguments",
    "DataMixin",
    # Utility functions
    "torch_load",
    "print_message",
    # Seed utilities
    "set_global_seed",
    "get_global_seed",
    "seed_worker",
    "dataloader_generator",
    "set_determinism",
]
