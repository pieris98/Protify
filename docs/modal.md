# Cloud Compute

This page documents how to run Protify jobs on cloud GPUs.

---

## Synthyra API (recommended)

The simplest way to run Protify on cloud GPUs is through the Synthyra API. Get your API key at [synthyra.com](https://synthyra.com).

**CLI:**
```bash
python -m main --synthyra_api_key YOUR_KEY --model_names ESM2-8 --data_names EC
```

**GUI:** Enter your Synthyra API key in the **Info** tab, configure your run, and click **Submit Remote Run**.

No Modal account, local GPU, or cloud provider setup required. Jobs are billed through your Synthyra account.

---

## Modal (advanced / self-hosted)

For users who want to manage their own Modal compute directly, Protify also supports running jobs on [Modal](https://modal.com). There are two options:

1. **Tk GUI + modal_backend.py**
   Use the **Modal** tab in the Tk GUI to deploy the backend, submit the current configuration as a remote job, poll status, and fetch logs/results/plots.

2. **Legacy: protify_modal_app.py (Gradio)**
   Deprecated in favor of the GUI + backend flow.

Both use the same config schema as CLI/YAML; the container runs the same `main.py` with path values rewritten for `/data/...`.

---

## How it works (modal_backend.py)

- **Deploy:** The GUI (or CLI) runs `modal deploy src/protify/modal_backend.py` (or equivalent). This builds the Modal image (Debian, Python 3.10, requirements.txt, src/) and deploys the app.
- **Submit:** The GUI sends the current `full_args` as a config dict to a Modal GPU function (e.g. `run_protify_job_a10`). The backend writes the config to a temp YAML, sets `PROTIFY_JOB_ID`, and runs `python -m main --yaml_path ...` from the container's Protify directory with `PYTHONPATH` set so that `main` is the same as local.
- **Paths:** Config paths (log_dir, results_dir, model_save_dir, embedding_save_dir, plots_dir, download_dir) are rewritten to `/data/logs`, `/data/results`, etc. A Modal Volume is mounted at `/data` for persistence.
- **Status:** The backend updates a job status file on the volume (e.g. PENDING, RUNNING, SUCCESS, FAILED) and optional heartbeat. The GUI polls a CPU function that reads this file.
- **Fetch:** A CPU function reads logs, results TSV, and plot files from the volume and returns them (e.g. base64 for images) so the GUI can save them locally (e.g. to `modal_artifacts`).

---

## Config schema

The config dict passed to the Modal GPU function has the same keys as [base.yaml](cli_and_config.md) and the CLI: ID (hf_username, hf_token, wandb_api_key, synthyra_api_key), paths, data (delimiter, max_length, trim, data_names, data_dirs, etc.), model (model_names or model_paths/model_types), embedding args, probe args (including lora), trainer args, ProteinGym and scikit options. Optional keys like replay_path and pretrained_probe_path can be set in the backend if missing. Paths under `data/` or `local_data/` and `*_dir` keys are normalized to the container layout.

---

## Path rewriting

In the container, default dirs are:

- `log_dir`: `/data/logs`
- `results_dir`: `/data/results`
- `plots_dir`: `/data/plots`
- `model_save_dir`: `/data/weights`
- `embedding_save_dir`: `/data/embeddings`
- `download_dir`: `/data/downloads`

User paths that point under local `data/` or `local_data/` are rewritten to `/data/...` so that the same config can be used locally and on Modal when data is uploaded or mounted.

---

## Legacy Gradio app (protify_modal_app.py)

- **Location:** [modal/protify_modal_app.py](../modal/protify_modal_app.py).
- **Entry:** A Gradio interface is served via `web_interface()` (ASGI app). Users configure the run via form fields; `form_values_to_config` builds a config dict that is passed to a GPU function (e.g. `run_protify_job_h200`, `run_protify_job_a10`). That function calls `_execute_protify_job(config, hf_token, job_id)`, which writes a temp config YAML and runs `python main.py --yaml_path <path>` from `/root/src/protify` with `PYTHONPATH=/root/src`.
- **Volume:** `modal.Volume.from_name("protify-data")` at `/data` for logs, results, plots, job_storage.json.
- **Get results:** `get_results(job_id)` reads from `/data/results`, `/data/plots`, `/data/logs` by job_id and returns file contents and base64 images. `list_jobs()` lists job IDs from the results directory.

The recommended approach is to use the Tk GUI and modal_backend.py instead of deploying this Gradio app.

---

## Deploy and run (recommended)

1. Install Modal: `pip install modal`.
2. Authenticate: set `modal_token_id` and `modal_token_secret` (or `modal_api_key` as `token_id:token_secret`).
3. In the Tk GUI **Info** tab, enter Modal credentials and paths.
4. In the **Modal** tab, set the Modal app name and backend path (e.g. `src/protify/modal_backend.py`), choose GPU type and timeout, then click **Deploy Modal Backend**.
5. Configure data, models, probe, and trainer in other tabs; click **Submit Remote Run**.
6. Use **Poll Status** (or auto polling) to see job status and optional log tail.
7. When the job completes, click **Fetch Logs/Results/Plots** to download artifacts to your local directory (e.g. `modal_artifacts`).

Pass Hugging Face and W&B tokens in the Info tab so the remote run can access gated models and log to W&B.

---

## See also

- [GUI](gui.md) for the Modal tab and background tasks
- [Configuration](cli_and_config.md) for options that are sent in the config
- [Getting started](getting_started.md) for local entry points
