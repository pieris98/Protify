"""Top-level package for Protify.

Exposes common subpackages for convenient imports whether Protify is used
as a submodule (installed or added to PYTHONPATH) or executed from source.
"""

# Re-export commonly used modules/APIs
from . import probes  # noqa: F401

__all__ = [
    "probes",
]


