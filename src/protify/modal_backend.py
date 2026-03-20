"""Protify Modal backend for training probes.

Image is built once at deploy time. Each job spawns a GPU sandbox that
reuses the pre-built image with per-task volumes.

Deploy:
    cd submodules/protify
    py -m modal deploy src/protify/modal_backend.py --name protify-backend
"""
import os
import sys
import time
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROTIFY_SRC = Path(__file__).resolve().parent        # .../protify/src/protify
PROTIFY_PKG = PROTIFY_SRC.parent                     # .../protify/src
REPO_ROOT = PROTIFY_PKG.parents[2]                    # synth repo root

# ---------------------------------------------------------------------------
# Image (built once at deploy time)
# ---------------------------------------------------------------------------

_requirements = REPO_ROOT / "requirements.txt"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pip", "setuptools")
    .pip_install_from_requirements(str(_requirements))
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("lightgbm", "pyfiglet")
    .env({
        "PYTHONPATH": "/root/src/protify:/root/src",
        "TOKENIZERS_PARALLELISM": "true",
    })
    .add_local_dir(
        str(PROTIFY_PKG),
        remote_path="/root/src",
        ignore=["__pycache__", "*.pyc", ".git", ".cache"],
    )
    .add_local_dir(
        str(REPO_ROOT / "core" / "models"),
        remote_path="/root/src/models",
        ignore=["__pycache__", "*.pyc"],
    )
    .add_local_file(
        str(REPO_ROOT / "entrypoint_setup.py"),
        remote_path="/root/src/entrypoint_setup.py",
    )
)

app = modal.App("protify-backend", image=image)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_MOUNT = "/embeddings"
DATA_MOUNT = "/data"
DEFAULT_EMBED_VOLUME = "protify-embeddings"
DEFAULT_DATA_VOLUME = "protify-data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_entry_script(yaml_str: str, task_name: str) -> str:
    """Build the Python script that runs inside the GPU sandbox."""
    config_path = DATA_MOUNT + "/configs/" + task_name + ".yaml"

    # Runner script written into protify dir so sys.path[0] = protify dir.
    # Registers InterpNet, then imports and runs main().
    runner_code = textwrap.dedent("""\
        import sys
        from models.interpnet import register_with_protify
        register_with_protify()
        sys.argv = ["main.py", "--yaml_path", "{config_path}"]
        from main import parse_arguments, main
        args = parse_arguments()
        result = main(args)
        sys.exit(result if result else 0)
    """).replace("{config_path}", config_path)

    return textwrap.dedent("""\
        import os, sys, subprocess

        # Write YAML config
        os.makedirs("{data_mount}/configs", exist_ok=True)
        config_path = "{config_path}"
        with open(config_path, "w") as f:
            f.write('''{yaml_str}''')

        # Write runner into protify dir
        protify_dir = "/root/src/protify"
        runner_path = os.path.join(protify_dir, "_oracle_runner.py")
        with open(runner_path, "w") as f:
            f.write('''{runner_code}''')

        result = subprocess.run([sys.executable, "-u", runner_path], cwd=protify_dir)
        if os.path.exists(runner_path):
            os.remove(runner_path)
        sys.exit(result.returncode)
    """).replace("{data_mount}", DATA_MOUNT).replace(
        "{config_path}", config_path
    ).replace("{yaml_str}", yaml_str).replace("{runner_code}", runner_code)


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

@app.function(timeout=86400)
def submit_protify_job(
    config: Dict[str, Any],
    gpu_type: str = "A10G",
    hf_token: Optional[str] = None,
    timeout_seconds: int = 86400,
    embed_volume_name: Optional[str] = None,
    data_volume_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a single probe. Spawns a GPU sandbox with per-task volumes."""
    import yaml

    task_name = config.get("push_raw_probe_repo", "unknown").split("/")[-1]
    job_id = task_name + "_" + str(int(time.time()))

    # Override paths to volume mounts
    config["embedding_save_dir"] = EMBED_MOUNT
    config["log_dir"] = DATA_MOUNT + "/logs"
    config["results_dir"] = DATA_MOUNT + "/results"
    config["model_save_dir"] = DATA_MOUNT + "/weights"
    config["plots_dir"] = DATA_MOUNT + "/plots"
    config["download_dir"] = DATA_MOUNT + "/downloads"

    if hf_token:
        config["hf_token"] = hf_token

    yaml_str = yaml.dump(config, default_flow_style=False)
    entry_script = _build_entry_script(yaml_str, task_name)

    # Create per-task volumes
    embed_name = embed_volume_name or DEFAULT_EMBED_VOLUME
    data_name = data_volume_name or DEFAULT_DATA_VOLUME
    embed_vol = modal.Volume.from_name(embed_name, create_if_missing=True)
    data_vol = modal.Volume.from_name(data_name, create_if_missing=True)

    # Build secrets
    env_secrets = {}
    if hf_token:
        env_secrets["HF_TOKEN"] = hf_token
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        env_secrets["WANDB_API_KEY"] = wandb_key
    secrets = [modal.Secret.from_dict(env_secrets)] if env_secrets else []

    print(f"[{task_name}] Spawning sandbox: gpu={gpu_type}, embed={embed_name}, data={data_name}")

    # Spawn GPU sandbox with pre-built image
    sb = modal.Sandbox.create(
        image=image,
        app=app,
        gpu=gpu_type,
        timeout=timeout_seconds,
        volumes={
            EMBED_MOUNT: embed_vol,
            DATA_MOUNT: data_vol,
        },
        secrets=secrets,
        workdir="/root/src",
    )

    sb.set_tags({"task": task_name, "job_id": job_id})

    # Write and execute entry script
    f = sb.open("/tmp/_entry.py", "w")
    f.write(entry_script)
    f.close()

    proc = sb.exec("python", "-u", "/tmp/_entry.py")

    # Stream output
    for line in proc.stdout:
        print(line, end="")
    for line in proc.stderr:
        print(line, end="", file=sys.stderr)

    proc.wait()
    exit_code = proc.returncode
    sb.terminate()

    status = "completed" if exit_code == 0 else "failed"
    print(f"[{task_name}] {status} (exit_code={exit_code})")

    return {
        "job_id": job_id,
        "task": task_name,
        "exit_code": exit_code,
        "status": status,
    }


@app.function(timeout=86400)
def run_sequential(
    jobs: List[Dict[str, Any]],
    gpu_type: str = "A10G",
    hf_token: Optional[str] = None,
    timeout_seconds: int = 86400,
) -> List[Dict[str, Any]]:
    """Run multiple training jobs sequentially."""
    results = []
    for job in jobs:
        task_name = job["name"]
        config = job["config"]
        embed_vol = config.pop("embed_volume_name", None)
        data_vol = config.pop("data_volume_name", None)

        print(f"\n{'=' * 60}")
        print(f"Starting: {task_name}")

        result = submit_protify_job.local(
            config=config,
            gpu_type=gpu_type,
            hf_token=hf_token,
            timeout_seconds=timeout_seconds,
            embed_volume_name=embed_vol,
            data_volume_name=data_vol,
        )
        results.append(result)
        print(f"Completed: {task_name} -> {result['status']}")

    return results
