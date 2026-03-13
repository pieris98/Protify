"""
Backend-only Modal app for GUI-driven Protify workflows.

This module intentionally avoids browser UI dependencies. It exposes remote
functions that the local Tk GUI can call to deploy, submit jobs, monitor status,
cancel jobs, and fetch artifacts.
"""

import base64
import glob
import json
import os
import random
import shutil
import sqlite3
import string
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import modal
import yaml


SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parents[1] if len(SCRIPT_DIR.parents) > 1 else SCRIPT_DIR

APP_NAME = "protify-backend"
PROTIFY_DEFAULT_GPU = "H100"
AVAILABLE_GPUS = ["H200", "H100", "A100-80GB", "A100", "L40S", "A10", "L4", "T4"]

GPU_CPU_MIN, GPU_CPU_MAX = 16.0, 16.0
GPU_MEMORY_MIN, GPU_MEMORY_MAX = 131072, 131072
MAX_CONTAINERS_GPU = 8
CPU_MEMORY_MIN, CPU_MEMORY_MAX = 4096, 8192
CPU_COUNT_MIN, CPU_COUNT_MAX = 2.0, 4.0
MAX_CONTAINERS_CPU = 10
SCALEDOWN_WINDOW_GPU = 10
SCALEDOWN_WINDOW_CPU = 300
TIMEOUT_SECONDS = 86400
HEARTBEAT_SECONDS = 10

STATUS_FILE_PATH = "/data/job_status.json"
LOG_DIR_DEFAULT = "/data/logs"
RESULTS_DIR_DEFAULT = "/data/results"
PLOTS_DIR_DEFAULT = "/data/plots"
WEIGHTS_DIR_DEFAULT = "/data/weights"
EMBED_DIR_DEFAULT = "/data/embeddings"
DOWNLOAD_DIR_DEFAULT = "/data/downloads"


def _build_image():
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git", "wget", "curl")
        .run_commands("pip install --upgrade pip setuptools")
    )

    req_file_path = "requirements.txt"
    if (PROJECT_ROOT / req_file_path).exists():
        image = image.add_local_file(req_file_path, "/tmp/requirements.txt", copy=True)
        image = image.run_commands("pip install -r /tmp/requirements.txt")
    else:
        image = image.run_commands("pip install torch transformers datasets")

    src_dir_path = "src"
    if (PROJECT_ROOT / src_dir_path).exists():
        image = image.add_local_dir(src_dir_path, "/root/src", copy=True)

    fastplms_dir = "src/protify/fastplms"
    if (PROJECT_ROOT / fastplms_dir).exists():
        image = image.add_local_dir(fastplms_dir, "/root/src/protify/fastplms", copy=True)

    readme_file_path = "README.md"
    if (PROJECT_ROOT / readme_file_path).exists():
        image = image.add_local_file(readme_file_path, "/root/README.md", copy=True)

    image = image.env(
        {
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "TF_ENABLE_ONEDNN_OPTS": "0",
            "TOKENIZERS_PARALLELISM": "true",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    return image


app = modal.App(APP_NAME)
image = _build_image()
volume = modal.Volume.from_name("protify-data", create_if_missing=True)

_status_lock = threading.Lock()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_job_id() -> str:
    random_letters = "".join(random.choices(string.ascii_uppercase, k=4))
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    return f"{date_str}_{random_letters}"


def _safe_read_json(json_path: str) -> Dict[str, Any]:
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def _safe_write_json(json_path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _update_job_status(job_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    with _status_lock:
        status_store = _safe_read_json(STATUS_FILE_PATH)
        if job_id not in status_store:
            status_store[job_id] = {
                "job_id": job_id,
                "status": "PENDING",
                "phase": "created",
                "created_at_utc": _now_utc_iso(),
                "updated_at_utc": _now_utc_iso(),
            }
        status_store[job_id].update(patch)
        status_store[job_id]["updated_at_utc"] = _now_utc_iso()
        _safe_write_json(STATUS_FILE_PATH, status_store)
        volume.commit()
        return status_store[job_id]


def _infer_phase_from_line(line: str, current_phase: str) -> str:
    lowered = line.lower()
    if "loading and preparing datasets" in lowered or "getting data" in lowered:
        return "data_loading"
    if "computing embeddings" in lowered or "saving embeddings" in lowered or "download embeddings" in lowered:
        return "embedding"
    if "starting training" in lowered or "training probe" in lowered or "run_wandb_hyperopt" in lowered:
        return "training"
    if "proteingym" in lowered:
        return "proteingym"
    if "generating visualization plots" in lowered:
        return "plotting"
    if "successfully saved model to huggingface hub" in lowered:
        return "pushing_to_hub"
    return current_phase


def _tail_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _fix_paths(config_obj: Any) -> Any:
    if isinstance(config_obj, dict):
        for key in list(config_obj.keys()):
            value = config_obj[key]
            if isinstance(value, str):
                if value.startswith("data/") or value.startswith("local_data/"):
                    config_obj[key] = f"/data/{value.split('/', 1)[1]}"
                elif key.endswith("_dir") and (not os.path.isabs(value)):
                    config_obj[key] = f"/data/{value}"
            elif isinstance(value, list):
                config_obj[key] = [_fix_paths(item) for item in value]
            elif isinstance(value, dict):
                config_obj[key] = _fix_paths(value)
    elif isinstance(config_obj, list):
        return [_fix_paths(item) for item in config_obj]
    return config_obj


def _prepare_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config_copy = dict(config)
    config_copy = _fix_paths(config_copy)

    if ("log_dir" not in config_copy) or (not config_copy["log_dir"]):
        config_copy["log_dir"] = LOG_DIR_DEFAULT
    if ("results_dir" not in config_copy) or (not config_copy["results_dir"]):
        config_copy["results_dir"] = RESULTS_DIR_DEFAULT
    if ("model_save_dir" not in config_copy) or (not config_copy["model_save_dir"]):
        config_copy["model_save_dir"] = WEIGHTS_DIR_DEFAULT
    if ("embedding_save_dir" not in config_copy) or (not config_copy["embedding_save_dir"]):
        config_copy["embedding_save_dir"] = EMBED_DIR_DEFAULT
    if ("plots_dir" not in config_copy) or (not config_copy["plots_dir"]):
        config_copy["plots_dir"] = PLOTS_DIR_DEFAULT
    if ("download_dir" not in config_copy) or (not config_copy["download_dir"]):
        config_copy["download_dir"] = DOWNLOAD_DIR_DEFAULT
    if "replay_path" not in config_copy:
        config_copy["replay_path"] = None
    if "pretrained_probe_path" not in config_copy:
        config_copy["pretrained_probe_path"] = None
    if "hf_home" not in config_copy:
        config_copy["hf_home"] = None

    path_keys = ["log_dir", "results_dir", "model_save_dir", "embedding_save_dir", "plots_dir", "download_dir"]
    for path_key in path_keys:
        os.makedirs(config_copy[path_key], exist_ok=True)

    return config_copy


def _run_protify_subprocess(
    prepared_config: Dict[str, Any],
    config_path: str,
    active_hf_token: Optional[str],
    wandb_api_key: Optional[str],
    synthyra_api_key: Optional[str],
    process_env: Dict[str, str],
    log_file_path: str,
    job_id: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    """Run main.py as a subprocess with streaming output and heartbeat monitoring.

    Returns a dict with keys: success, return_code, stdout, stderr, timed_out.
    """
    config_to_dump = dict(prepared_config)
    config_to_dump["hf_token"] = None
    config_to_dump["wandb_api_key"] = None
    config_to_dump["synthyra_api_key"] = None
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.dump(config_to_dump, config_file, default_flow_style=False, allow_unicode=True, sort_keys=False)

    command = ["python", "-u", "main.py", "--yaml_path", str(config_path)]
    if active_hf_token is not None:
        command.extend(["--hf_token", active_hf_token])
    if wandb_api_key is not None:
        command.extend(["--wandb_api_key", wandb_api_key])
    if synthyra_api_key is not None:
        command.extend(["--synthyra_api_key", synthyra_api_key])

    stdout_lines = []
    stderr_lines = []
    log_lock = threading.Lock()
    phase_state = {"phase": "startup"}

    def append_log(log_line: str) -> None:
        with log_lock:
            with open(log_file_path, "a", encoding="utf-8", errors="ignore") as log_file:
                log_file.write(log_line + "\n")

    def stream_output(pipe, output_list, prefix: str = ""):
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    continue
                clean_line = line.rstrip("\n")
                full_line = f"{prefix}{clean_line}"
                output_list.append(full_line)
                phase_state["phase"] = _infer_phase_from_line(clean_line, phase_state["phase"])
                append_log(full_line)
                print(full_line, flush=True)
        finally:
            pipe.close()

    timed_out = False
    process = None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd="/root/src/protify",
            env=process_env,
        )

        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, stdout_lines, ""), daemon=True)
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, stderr_lines, "[STDERR] "), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        max_runtime_seconds = timeout_seconds
        if max_runtime_seconds > TIMEOUT_SECONDS - 60:
            max_runtime_seconds = TIMEOUT_SECONDS - 60

        start_time = time.time()
        last_heartbeat = 0.0
        while process.poll() is None:
            now = time.time()
            if now - start_time > max_runtime_seconds:
                timed_out = True
                process.kill()
                break
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                _update_job_status(
                    job_id,
                    {
                        "status": "RUNNING",
                        "phase": phase_state["phase"],
                        "last_heartbeat_utc": _now_utc_iso(),
                    },
                )
                last_heartbeat = now
            time.sleep(1)

        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        return_code = process.returncode if process is not None else -1
        stdout_text = "\n".join(stdout_lines)
        stderr_text = "\n".join(stderr_lines)

        return {
            "success": not timed_out and return_code == 0,
            "return_code": return_code,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "timed_out": timed_out,
        }
    except Exception as error:
        return {
            "success": False,
            "return_code": -1,
            "stdout": "",
            "stderr": str(error),
            "timed_out": False,
        }


def _find_staging_db(staging_dir: str) -> Optional[str]:
    """Find the .db file in a staging directory."""
    matches = glob.glob(os.path.join(staging_dir, "*.db"))
    if len(matches) == 0:
        return None
    return matches[0]


def _execute_protify_job(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    synthyra_api_key: Optional[str] = None,
    job_id: Optional[str] = None,
    gpu_type: Optional[str] = None,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    if job_id is None:
        job_id = _generate_job_id()

    selected_gpu = gpu_type if gpu_type in AVAILABLE_GPUS else PROTIFY_DEFAULT_GPU
    _update_job_status(
        job_id,
        {
            "status": "RUNNING",
            "phase": "startup",
            "gpu_type": selected_gpu,
            "last_heartbeat_utc": _now_utc_iso(),
            "started_at_utc": _now_utc_iso(),
            "error": None,
        },
    )

    active_hf_token = hf_token
    if active_hf_token is None:
        active_hf_token = os.environ.get("HF_TOKEN")

    if active_hf_token is not None:
        try:
            from huggingface_hub import login

            os.environ["HF_TOKEN"] = active_hf_token
            login(active_hf_token)
        except Exception:
            pass

    prepared_config = _prepare_config(config)
    log_file_path = os.path.join(prepared_config["log_dir"], f"{job_id}.txt")
    _update_job_status(
        job_id,
        {
            "log_file_path": log_file_path,
            "results_dir": prepared_config["results_dir"],
            "plots_dir": prepared_config["plots_dir"],
        },
    )

    run_dir = Path("/tmp/protify_run") / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = str(run_dir / "config.yaml")

    process_env = os.environ.copy()
    process_env["PYTHONPATH"] = "/root/src"
    process_env["WORKING_DIR"] = "/root"
    process_env["PYTHONUNBUFFERED"] = "1"
    process_env["PROTIFY_JOB_ID"] = job_id
    process_env["CUDA_VISIBLE_DEVICES"] = "0"
    if active_hf_token is not None:
        process_env["HF_TOKEN"] = active_hf_token
    if wandb_api_key is not None:
        process_env["WANDB_API_KEY"] = wandb_api_key

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"[{_now_utc_iso()}] Starting job {job_id}\n")
        log_file.write(f"GPU={selected_gpu}\n")
    volume.commit()

    staging_merge = config.get("staging_merge_enabled", False)
    task_name = config.get("staging_task_name", job_id)
    golden_embed_dir = prepared_config["embedding_save_dir"]

    def _fail(error_msg: str, stdout: str = "") -> Dict[str, Any]:
        _update_job_status(
            job_id,
            {
                "status": "FAILED",
                "phase": "failed",
                "last_heartbeat_utc": _now_utc_iso(),
                "error": _tail_text(error_msg, 5000),
                "exit_code": -1,
                "finished_at_utc": _now_utc_iso(),
            },
        )
        return {
            "success": False,
            "job_id": job_id,
            "status": "FAILED",
            "error": _tail_text(error_msg, 5000),
            "stdout": _tail_text(stdout, 5000),
        }

    try:
        if staging_merge:
            # Phase 1: Embed to per-task staging directory
            staging_embed_dir = os.path.join(golden_embed_dir, "staging", task_name)
            os.makedirs(staging_embed_dir, exist_ok=True)

            embed_config = dict(prepared_config)
            embed_config["embedding_save_dir"] = staging_embed_dir

            _update_job_status(job_id, {"phase": "embedding"})
            process_env["_PROTIFY_EMBED_PHASE"] = "1"
            embed_result = _run_protify_subprocess(
                prepared_config=embed_config,
                config_path=config_path,
                active_hf_token=active_hf_token,
                wandb_api_key=wandb_api_key,
                synthyra_api_key=synthyra_api_key,
                process_env=process_env,
                log_file_path=log_file_path,
                job_id=job_id,
                timeout_seconds=timeout_seconds,
            )
            process_env.pop("_PROTIFY_EMBED_PHASE", None)
            if not embed_result["success"]:
                error_msg = embed_result["stderr"] if embed_result["stderr"] else "Embedding subprocess failed."
                if embed_result["timed_out"]:
                    error_msg = f"Embedding timed out after {timeout_seconds} seconds."
                return _fail(error_msg, embed_result["stdout"])

            # Phase 2: Merge staging DB into golden DB (serialized via max_containers=1)
            volume.commit()  # commit any newly embedded sequences so merge container can see them
            staging_db = _find_staging_db(staging_embed_dir)
            if staging_db is not None:
                golden_db = os.path.join(golden_embed_dir, os.path.basename(staging_db))
                _update_job_status(job_id, {"phase": "merging_embeddings"})
                merge_staging_embeddings.remote(staging_db, golden_db)
                volume.reload()  # make merged golden DB visible to training subprocess

            # Phase 3: Train (reads from golden DB, finds all seqs, no embedding needed)
            train_config = dict(prepared_config)
            train_config["embedding_save_dir"] = golden_embed_dir

            _update_job_status(job_id, {"phase": "training"})
            train_result = _run_protify_subprocess(
                prepared_config=train_config,
                config_path=config_path,
                active_hf_token=active_hf_token,
                wandb_api_key=wandb_api_key,
                synthyra_api_key=synthyra_api_key,
                process_env=process_env,
                log_file_path=log_file_path,
                job_id=job_id,
                timeout_seconds=timeout_seconds,
            )
            if not train_result["success"]:
                error_msg = train_result["stderr"] if train_result["stderr"] else "Training subprocess failed."
                if train_result["timed_out"]:
                    error_msg = f"Training timed out after {timeout_seconds} seconds."
                return _fail(error_msg, train_result["stdout"])

            _update_job_status(
                job_id,
                {
                    "status": "SUCCESS",
                    "phase": "completed",
                    "last_heartbeat_utc": _now_utc_iso(),
                    "error": None,
                    "exit_code": 0,
                    "finished_at_utc": _now_utc_iso(),
                },
            )
            return {
                "success": True,
                "job_id": job_id,
                "status": "SUCCESS",
                "stdout": _tail_text(train_result["stdout"], 5000),
            }

        else:
            # Original single-phase flow (no staging)
            result = _run_protify_subprocess(
                prepared_config=prepared_config,
                config_path=config_path,
                active_hf_token=active_hf_token,
                wandb_api_key=wandb_api_key,
                synthyra_api_key=synthyra_api_key,
                process_env=process_env,
                log_file_path=log_file_path,
                job_id=job_id,
                timeout_seconds=timeout_seconds,
            )

            if result["timed_out"]:
                _update_job_status(
                    job_id,
                    {
                        "status": "TIMEOUT",
                        "phase": "timeout",
                        "last_heartbeat_utc": _now_utc_iso(),
                        "error": f"Process timed out.",
                        "exit_code": -1,
                        "finished_at_utc": _now_utc_iso(),
                    },
                )
                return {
                    "success": False,
                    "job_id": job_id,
                    "status": "TIMEOUT",
                    "error": "Process timed out.",
                    "stdout": _tail_text(result["stdout"], 5000),
                }

            if not result["success"]:
                return _fail(
                    result["stderr"] if result["stderr"] else "Unknown subprocess error.",
                    result["stdout"],
                )

            _update_job_status(
                job_id,
                {
                    "status": "SUCCESS",
                    "phase": "completed",
                    "last_heartbeat_utc": _now_utc_iso(),
                    "error": None,
                    "exit_code": 0,
                    "finished_at_utc": _now_utc_iso(),
                },
            )
            return {
                "success": True,
                "job_id": job_id,
                "status": "SUCCESS",
                "stdout": _tail_text(result["stdout"], 5000),
            }

    except Exception as error:
        return _fail(str(error))


@app.function(
    image=image,
    volumes={"/data": volume},
    memory=(CPU_MEMORY_MIN, CPU_MEMORY_MAX),
    cpu=(CPU_COUNT_MIN, CPU_COUNT_MAX),
    max_containers=1,
    timeout=3600,
)
def merge_staging_embeddings(staging_db_path: str, golden_db_path: str) -> Dict[str, Any]:
    """Merge a per-task staging DB into the shared golden DB.

    max_containers=1 ensures Modal queues concurrent calls and runs them
    one at a time, preventing cross-container write contention on the golden DB.

    Uses Python-level deduplication to avoid O(N * B-tree depth) random page
    reads on the Modal network volume: load existing keys into a set (one
    sequential scan), stream staging rows, filter in Python, bulk-insert only
    new rows. PRAGMA journal_mode=MEMORY eliminates WAL file writes during the
    merge; WAL is restored at the end for subsequent readers.
    """
    volume.reload()
    os.makedirs(os.path.dirname(golden_db_path), exist_ok=True)
    conn = sqlite3.connect(golden_db_path, timeout=3600)
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-262144")  # 256 MB
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("CREATE TABLE IF NOT EXISTS embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)")
    conn.commit()

    existing_seqs = {row[0] for row in conn.execute("SELECT sequence FROM embeddings")}

    staging_conn = sqlite3.connect(staging_db_path, timeout=3600)
    staging_conn.execute("PRAGMA cache_size=-131072")  # 128 MB

    _CHUNK_SIZE = 5_000
    chunk: list = []
    inserted = 0
    for seq, emb in staging_conn.execute("SELECT sequence, embedding FROM embeddings"):
        if seq not in existing_seqs:
            chunk.append((seq, emb))
            if len(chunk) >= _CHUNK_SIZE:
                conn.executemany("INSERT INTO embeddings VALUES (?, ?)", chunk)
                conn.commit()
                inserted += len(chunk)
                chunk.clear()
    if chunk:
        conn.executemany("INSERT INTO embeddings VALUES (?, ?)", chunk)
        conn.commit()
        inserted += len(chunk)

    staging_conn.close()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.close()
    volume.commit()
    return {"success": True, "staging": staging_db_path, "golden": golden_db_path, "inserted": inserted}


@app.function(
    image=image,
    gpu="H200",
    volumes={"/data": volume},
    memory=(GPU_MEMORY_MIN, GPU_MEMORY_MAX),
    cpu=(GPU_CPU_MIN, GPU_CPU_MAX),
    max_containers=MAX_CONTAINERS_GPU,
    scaledown_window=SCALEDOWN_WINDOW_GPU,
    timeout=TIMEOUT_SECONDS,
)
def run_protify_job_h200(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    synthyra_api_key: Optional[str] = None,
    job_id: Optional[str] = None,
    gpu_type: Optional[str] = None,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, wandb_api_key, synthyra_api_key, job_id, gpu_type, timeout_seconds)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": volume},
    memory=(GPU_MEMORY_MIN, GPU_MEMORY_MAX),
    cpu=(GPU_CPU_MIN, GPU_CPU_MAX),
    max_containers=MAX_CONTAINERS_GPU,
    scaledown_window=SCALEDOWN_WINDOW_GPU,
    timeout=TIMEOUT_SECONDS,
)
def run_protify_job_h100(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    synthyra_api_key: Optional[str] = None,
    job_id: Optional[str] = None,
    gpu_type: Optional[str] = None,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, wandb_api_key, synthyra_api_key, job_id, gpu_type, timeout_seconds)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    memory=(GPU_MEMORY_MIN, GPU_MEMORY_MAX),
    cpu=(GPU_CPU_MIN, GPU_CPU_MAX),
    max_containers=MAX_CONTAINERS_GPU,
    scaledown_window=SCALEDOWN_WINDOW_GPU,
    timeout=TIMEOUT_SECONDS,
)
def run_protify_job_a100_80gb(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    synthyra_api_key: Optional[str] = None,
    job_id: Optional[str] = None,
    gpu_type: Optional[str] = None,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, wandb_api_key, synthyra_api_key, job_id, gpu_type, timeout_seconds)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": volume},
    memory=(GPU_MEMORY_MIN, GPU_MEMORY_MAX),
    cpu=(GPU_CPU_MIN, GPU_CPU_MAX),
    max_containers=MAX_CONTAINERS_GPU,
    scaledown_window=SCALEDOWN_WINDOW_GPU,
    timeout=TIMEOUT_SECONDS,
)
def run_protify_job_a100(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    synthyra_api_key: Optional[str] = None,
    job_id: Optional[str] = None,
    gpu_type: Optional[str] = None,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, wandb_api_key, synthyra_api_key, job_id, gpu_type, timeout_seconds)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/data": volume},
    memory=(GPU_MEMORY_MIN, GPU_MEMORY_MAX),
    cpu=(GPU_CPU_MIN, GPU_CPU_MAX),
    max_containers=MAX_CONTAINERS_GPU,
    scaledown_window=SCALEDOWN_WINDOW_GPU,
    timeout=TIMEOUT_SECONDS,
)
def run_protify_job_l40s(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    synthyra_api_key: Optional[str] = None,
    job_id: Optional[str] = None,
    gpu_type: Optional[str] = None,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, wandb_api_key, synthyra_api_key, job_id, gpu_type, timeout_seconds)


@app.function(
    image=image,
    gpu="A10",
    volumes={"/data": volume},
    memory=(GPU_MEMORY_MIN, GPU_MEMORY_MAX),
    cpu=(GPU_CPU_MIN, GPU_CPU_MAX),
    max_containers=MAX_CONTAINERS_GPU,
    scaledown_window=SCALEDOWN_WINDOW_GPU,
    timeout=TIMEOUT_SECONDS,
)
def run_protify_job_a10(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    synthyra_api_key: Optional[str] = None,
    job_id: Optional[str] = None,
    gpu_type: Optional[str] = None,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, wandb_api_key, synthyra_api_key, job_id, gpu_type, timeout_seconds)


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": volume},
    memory=(GPU_MEMORY_MIN, GPU_MEMORY_MAX),
    cpu=(GPU_CPU_MIN, GPU_CPU_MAX),
    max_containers=MAX_CONTAINERS_GPU,
    scaledown_window=SCALEDOWN_WINDOW_GPU,
    timeout=TIMEOUT_SECONDS,
)
def run_protify_job_l4(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    synthyra_api_key: Optional[str] = None,
    job_id: Optional[str] = None,
    gpu_type: Optional[str] = None,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, wandb_api_key, synthyra_api_key, job_id, gpu_type, timeout_seconds)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume},
    memory=(GPU_MEMORY_MIN, GPU_MEMORY_MAX),
    cpu=(GPU_CPU_MIN, GPU_CPU_MAX),
    max_containers=MAX_CONTAINERS_GPU,
    scaledown_window=SCALEDOWN_WINDOW_GPU,
    timeout=TIMEOUT_SECONDS,
)
def run_protify_job_t4(
    config: Dict[str, Any],
    hf_token: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    synthyra_api_key: Optional[str] = None,
    job_id: Optional[str] = None,
    gpu_type: Optional[str] = None,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    return _execute_protify_job(config, hf_token, wandb_api_key, synthyra_api_key, job_id, gpu_type, timeout_seconds)


gpu_functions = {
    "H200": run_protify_job_h200,
    "H100": run_protify_job_h100,
    "A100-80GB": run_protify_job_a100_80gb,
    "A100": run_protify_job_a100,
    "L40S": run_protify_job_l40s,
    "A10": run_protify_job_a10,
    "L4": run_protify_job_l4,
    "T4": run_protify_job_t4,
}


@app.function(
    image=image,
    volumes={"/data": volume},
    memory=(CPU_MEMORY_MIN, CPU_MEMORY_MAX),
    cpu=(CPU_COUNT_MIN, CPU_COUNT_MAX),
    max_containers=MAX_CONTAINERS_CPU,
    scaledown_window=SCALEDOWN_WINDOW_CPU,
)
def submit_protify_job(
    config: Dict[str, Any],
    gpu_type: str = PROTIFY_DEFAULT_GPU,
    hf_token: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    synthyra_api_key: Optional[str] = None,
    timeout_seconds: int = TIMEOUT_SECONDS,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    if job_id is None:
        job_id = _generate_job_id()

    selected_gpu = gpu_type if gpu_type in AVAILABLE_GPUS else PROTIFY_DEFAULT_GPU
    _update_job_status(
        job_id,
        {
            "status": "PENDING",
            "phase": "queued",
            "gpu_type": selected_gpu,
            "last_heartbeat_utc": _now_utc_iso(),
            "error": None,
        },
    )

    selected_gpu_function = gpu_functions[selected_gpu]
    handle = selected_gpu_function.spawn(
        config=config,
        hf_token=hf_token,
        wandb_api_key=wandb_api_key,
        synthyra_api_key=synthyra_api_key,
        job_id=job_id,
        gpu_type=selected_gpu,
        timeout_seconds=timeout_seconds,
    )
    function_call_id = handle.object_id
    _update_job_status(
        job_id,
        {
            "status": "RUNNING",
            "phase": "queued",
            "function_call_id": function_call_id,
            "last_heartbeat_utc": _now_utc_iso(),
        },
    )
    return {
        "success": True,
        "job_id": job_id,
        "function_call_id": function_call_id,
        "status": "RUNNING",
        "gpu_type": selected_gpu,
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    memory=(CPU_MEMORY_MIN, CPU_MEMORY_MAX),
    cpu=(CPU_COUNT_MIN, CPU_COUNT_MAX),
    max_containers=MAX_CONTAINERS_CPU,
    scaledown_window=SCALEDOWN_WINDOW_CPU,
)
def get_job_status(job_id: str) -> Dict[str, Any]:
    volume.reload()
    status_store = _safe_read_json(STATUS_FILE_PATH)
    if job_id not in status_store:
        return {"success": False, "job_id": job_id, "error": "Job ID not found."}

    job_status = status_store[job_id]
    heartbeat_age_seconds = None
    if "last_heartbeat_utc" in job_status and job_status["last_heartbeat_utc"]:
        try:
            heartbeat_time = datetime.fromisoformat(job_status["last_heartbeat_utc"])
            heartbeat_age_seconds = (datetime.now(timezone.utc) - heartbeat_time).total_seconds()
        except Exception:
            heartbeat_age_seconds = None
    job_status["heartbeat_age_seconds"] = heartbeat_age_seconds
    job_status["success"] = True
    return job_status


@app.function(
    image=image,
    volumes={"/data": volume},
    memory=(CPU_MEMORY_MIN, CPU_MEMORY_MAX),
    cpu=(CPU_COUNT_MIN, CPU_COUNT_MAX),
    max_containers=MAX_CONTAINERS_CPU,
    scaledown_window=SCALEDOWN_WINDOW_CPU,
)
def get_job_log_tail(job_id: str, max_chars: int = 5000) -> Dict[str, Any]:
    volume.reload()
    status_store = _safe_read_json(STATUS_FILE_PATH)
    status_entry = status_store[job_id] if job_id in status_store else None
    if status_entry is not None and "log_file_path" in status_entry and status_entry["log_file_path"]:
        log_file_path = status_entry["log_file_path"]
    else:
        log_file_path = os.path.join(LOG_DIR_DEFAULT, f"{job_id}.txt")

    if not os.path.exists(log_file_path):
        if status_entry is None:
            return {"success": False, "job_id": job_id, "error": "Job ID not found.", "log_tail": ""}
        return {"success": True, "job_id": job_id, "log_tail": ""}

    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as log_file:
        text = log_file.read()
    return {"success": True, "job_id": job_id, "log_tail": _tail_text(text, max_chars)}


@app.function(
    image=image,
    volumes={"/data": volume},
    memory=(CPU_MEMORY_MIN, CPU_MEMORY_MAX),
    cpu=(CPU_COUNT_MIN, CPU_COUNT_MAX),
    max_containers=MAX_CONTAINERS_CPU,
    scaledown_window=SCALEDOWN_WINDOW_CPU,
)
def get_job_log_delta(job_id: str, offset: int = 0, max_chars: int = 5000) -> Dict[str, Any]:
    volume.reload()
    if offset < 0:
        offset = 0
    if max_chars <= 0:
        max_chars = 1

    status_store = _safe_read_json(STATUS_FILE_PATH)
    status_entry = status_store[job_id] if job_id in status_store else None
    if status_entry is not None and "log_file_path" in status_entry and status_entry["log_file_path"]:
        log_file_path = status_entry["log_file_path"]
    else:
        log_file_path = os.path.join(LOG_DIR_DEFAULT, f"{job_id}.txt")

    if not os.path.exists(log_file_path):
        return {
            "success": True,
            "job_id": job_id,
            "file_exists": False,
            "chunk": "",
            "next_offset": offset,
            "file_size": 0,
        }

    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as log_file:
        text = log_file.read()
    file_size = len(text)
    if offset > file_size:
        offset = file_size
    end_offset = offset + max_chars
    if end_offset > file_size:
        end_offset = file_size
    chunk = text[offset:end_offset]
    return {
        "success": True,
        "job_id": job_id,
        "file_exists": True,
        "chunk": chunk,
        "next_offset": end_offset,
        "file_size": file_size,
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    memory=(CPU_MEMORY_MIN, CPU_MEMORY_MAX),
    cpu=(CPU_COUNT_MIN, CPU_COUNT_MAX),
    max_containers=MAX_CONTAINERS_CPU,
    scaledown_window=SCALEDOWN_WINDOW_CPU,
)
def delete_modal_embeddings() -> Dict[str, Any]:
    volume.reload()
    embedding_dir = Path(EMBED_DIR_DEFAULT)
    if not embedding_dir.exists():
        return {
            "success": True,
            "message": f"Embedding directory does not exist: {EMBED_DIR_DEFAULT}",
            "deleted_files": 0,
            "deleted_dirs": 0,
        }

    deleted_files = 0
    deleted_dirs = 0
    for path in embedding_dir.glob("*"):
        if path.is_file():
            path.unlink()
            deleted_files += 1
        elif path.is_dir():
            shutil.rmtree(path)
            deleted_dirs += 1

    volume.commit()
    return {
        "success": True,
        "message": f"Deleted modal embedding cache contents ({deleted_files} files, {deleted_dirs} directories).",
        "deleted_files": deleted_files,
        "deleted_dirs": deleted_dirs,
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    memory=(CPU_MEMORY_MIN, CPU_MEMORY_MAX),
    cpu=(CPU_COUNT_MIN, CPU_COUNT_MAX),
    max_containers=MAX_CONTAINERS_CPU,
    scaledown_window=SCALEDOWN_WINDOW_CPU,
)
def cancel_protify_job(function_call_id: str, job_id: Optional[str] = None) -> Dict[str, Any]:
    function_call = modal.FunctionCall.from_id(function_call_id)
    function_call.cancel()
    if job_id is not None:
        _update_job_status(
            job_id,
            {
                "status": "TERMINATED",
                "phase": "cancelled",
                "last_heartbeat_utc": _now_utc_iso(),
                "finished_at_utc": _now_utc_iso(),
            },
        )
    return {"success": True, "function_call_id": function_call_id, "job_id": job_id}


@app.function(
    image=image,
    volumes={"/data": volume},
    memory=(CPU_MEMORY_MIN, CPU_MEMORY_MAX),
    cpu=(CPU_COUNT_MIN, CPU_COUNT_MAX),
    max_containers=MAX_CONTAINERS_CPU,
    scaledown_window=SCALEDOWN_WINDOW_CPU,
)
def get_results(job_id: str) -> Dict[str, Any]:
    volume.reload()
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"}
    results = {"success": True, "files": {}, "images": {}}

    results_dir = Path(RESULTS_DIR_DEFAULT)
    plots_dir = Path(PLOTS_DIR_DEFAULT)
    logs_dir = Path(LOG_DIR_DEFAULT)

    collected_files = set()
    result_file = results_dir / f"{job_id}.tsv"
    if result_file.exists():
        collected_files.add(result_file)
    log_file = logs_dir / f"{job_id}.txt"
    if log_file.exists():
        collected_files.add(log_file)
    plot_dir = plots_dir / job_id
    if plot_dir.exists() and plot_dir.is_dir():
        for file_path in plot_dir.rglob("*"):
            if file_path.is_file():
                collected_files.add(file_path)

    for file_path in collected_files:
        relative_path = str(file_path.relative_to(Path("/data")))
        suffix = file_path.suffix.lower()
        try:
            if suffix in image_extensions:
                with open(file_path, "rb") as image_file:
                    encoded = base64.b64encode(image_file.read()).decode("utf-8")
                mime_type = f"image/{suffix[1:]}" if suffix != ".svg" else "image/svg+xml"
                results["images"][relative_path] = {"data": encoded, "mime_type": mime_type}
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as text_file:
                    results["files"][relative_path] = text_file.read()
        except Exception as error:
            if suffix in image_extensions:
                results["images"][relative_path] = {"error": str(error)}
            else:
                results["files"][relative_path] = f"Error reading file: {error}"

    return results


@app.function(
    image=image,
    volumes={"/data": volume},
    memory=(CPU_MEMORY_MIN, CPU_MEMORY_MAX),
    cpu=(CPU_COUNT_MIN, CPU_COUNT_MAX),
    max_containers=MAX_CONTAINERS_CPU,
    scaledown_window=SCALEDOWN_WINDOW_CPU,
)
def list_jobs() -> Dict[str, Any]:
    volume.reload()
    status_store = _safe_read_json(STATUS_FILE_PATH)
    jobs = []
    for job_id in status_store:
        jobs.append(status_store[job_id])
    jobs.sort(key=lambda item: item["job_id"], reverse=True)
    return {"success": True, "jobs": jobs}


if __name__ == "__main__":
    with app.run():
        pass
