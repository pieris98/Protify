"""Benchmark modules for Protify.

This file ensures that the `benchmarks` directory is treated as a Python
package across execution contexts (e.g., running from `src/protify` with
`python -m main` or from project root with `python -m protify.main`).
"""

__all__ = [
    "proteingym",
]


