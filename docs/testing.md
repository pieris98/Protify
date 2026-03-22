# Testing

This page describes the Protify testing suite: where tests live, what they cover, and how to run them. For running tests in Docker when a Dockerfile exists, see the note at the end.

---

## Overview

Tests are under [src/protify/testing_suite/](https://github.com/gleghorn-lab/Protify/blob/main/src/protify/testing_suite/). They cover modal utilities, lazy-predict integration, packaged probe export, probe attention behavior, and embedding pipeline behavior. Run them with pytest from the repository root (or from `src/protify` with the appropriate Python path).

---

## Test layout

| File | Description |
|------|-------------|
| **test_modal_utils.py** | Tests for Modal-related utilities (e.g. parsing Modal API key, env handling). |
| **test_lazy_predict.py** | Tests for the lazy_predict / scikit integration. |
| **test_packaged_probe_export.py** | Tests for exporting and loading packaged probe models (probe type inference, config round-trip). |
| **test_probe_attention.py** | Tests for probe attention components (e.g. attention backends, masking). |
| **embedding_test.py** | Tests for the embedding pipeline (embedder, filenames, storage). |
| **__init__.py** | Package marker. |

Additional tests may exist in the same directory or under other test directories; the list above reflects the current testing_suite contents.

---

## How to run

From the **repository root** (so that `src` is on the path):

```bash
py -m pytest src/protify/testing_suite -v
```

To run a single file:

```bash
py -m pytest src/protify/testing_suite/test_modal_utils.py -v
```

To run with coverage (if you have pytest-cov):

```bash
py -m pytest src/protify/testing_suite -v --cov=src.protify --cov-report=term-missing
```

On Windows use `py`; on Linux/mac you can use `python` if preferred. Ensure the project dependencies are installed (`pip install -r requirements.txt`).

---

## Docker

If the project has a [Dockerfile](https://github.com/gleghorn-lab/Protify/blob/main/Dockerfile) and you want tests to run in a container (e.g. with a specific CUDA or OS environment), build the image first, then run pytest with the workspace mounted and working directory at repo root (`-w /workspace`):

**Linux / macOS:**
```bash
docker build -t protify-env:latest .
docker run --rm -v "${PWD}":/workspace -w /workspace protify-env:latest python -m pytest src/protify/testing_suite -v
```

**Windows:**
```bash
docker build -t protify-env:latest .
docker run --rm -v "%CD%":/workspace -w /workspace protify-env:latest py -m pytest src/protify/testing_suite -v
```

For GPU-dependent tests, add `--gpus all` to the run command. To run the main CLI or GUI in Docker, use `-w /workspace/src/protify` and `python -m main` or `python -m gui`; see [Getting started > Docker](getting_started.md#docker).

---

## See also

- [Getting started](getting_started.md) for installation and entry points
- [Probes and training](probes_and_training.md) for packaged probe export
- [Models and embeddings](models_and_embeddings.md) for embedder behavior
