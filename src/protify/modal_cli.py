import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _is_json_scalar(value):
    return value is None or isinstance(value, (bool, int, float, str))


def _to_json_safe(value):
    if _is_json_scalar(value):
        return value
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, dict):
        converted = {}
        for key in value:
            converted[key] = _to_json_safe(value[key])
        return converted
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _should_auto_run_modal(args):
    if "PROTIFY_JOB_ID" in os.environ and os.environ["PROTIFY_JOB_ID"] != "":
        return False
    if args.replay_path is not None:
        return False
    if not args.modal_cli_credentials_provided:
        return False
    return args.modal_token_id is not None and args.modal_token_secret is not None


def _modal_subprocess_env(args):
    env = os.environ.copy()
    env["MODAL_TOKEN_ID"] = args.modal_token_id
    env["MODAL_TOKEN_SECRET"] = args.modal_token_secret
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _deploy_modal_backend(args):
    repo_root = _repo_root()
    backend_path = repo_root / "src" / "protify" / "modal_backend.py"
    assert backend_path.exists(), f"Modal backend not found at {backend_path}"
    app_name = "protify-backend"
    env = _modal_subprocess_env(args)

    primary_command = [sys.executable, "-m", "modal", "deploy", str(backend_path), "--name", app_name]
    try:
        process = subprocess.run(
            primary_command,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        fallback_command = ["modal", "deploy", str(backend_path), "--name", app_name]
        process = subprocess.run(
            fallback_command,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    if process.returncode != 0:
        stderr_text = process.stderr if process.stderr is not None else ""
        stdout_text = process.stdout if process.stdout is not None else ""
        combined_output = f"{stdout_text}\n{stderr_text}".strip()
        if "No module named modal" in combined_output:
            raise RuntimeError("Modal is not installed in this Python environment. Install it with: py -m pip install modal")
        raise RuntimeError(f"Modal deploy failed:\n{combined_output}")

    stdout_text = process.stdout if process.stdout is not None else ""
    if stdout_text:
        print(stdout_text[-4000:])


def _build_modal_config_from_args(args):
    config = {}
    excluded_keys = {
        "modal_token_id",
        "modal_token_secret",
        "modal_api_key",
        "modal_cli_credentials_provided",
        "rebuild_modal",
        "delete_modal_embeddings",
    }
    for key in args.__dict__:
        if key in excluded_keys:
            continue
        config[key] = _to_json_safe(args.__dict__[key])
    config["replay_path"] = None
    return config


def _save_modal_artifacts(result_payload, output_root, job_id):
    output_root_path = Path(output_root)
    job_dir = output_root_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    files_payload = result_payload["files"] if "files" in result_payload else {}
    for rel_path in files_payload:
        local_path = job_dir / Path(rel_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as file:
            file.write(files_payload[rel_path])

    images_payload = result_payload["images"] if "images" in result_payload else {}
    for rel_path in images_payload:
        image_info = images_payload[rel_path]
        if "data" not in image_info:
            continue
        local_path = job_dir / Path(rel_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        image_bytes = base64.b64decode(image_info["data"])
        with open(local_path, "wb") as file:
            file.write(image_bytes)

    summary_path = job_dir / "modal_fetch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(result_payload, file, indent=2)
    return str(job_dir)


def _coerce_modal_terminal_payload(remote_result):
    if isinstance(remote_result, dict):
        payload = dict(remote_result)
        if "status" not in payload:
            if "success" in payload and payload["success"]:
                payload["status"] = "SUCCESS"
            elif "success" in payload and not payload["success"]:
                payload["status"] = "FAILED"
            else:
                payload["status"] = "SUCCESS"
        return payload
    return {"status": "SUCCESS"}


def _run_on_modal_cli(args):
    try:
        import modal
    except Exception as error:
        raise RuntimeError("Modal SDK is required for CLI remote execution. Install with: py -m pip install modal") from error

    app_name = "protify-backend"
    gpu_type = "A10"
    if "modal_gpu_type" in args.__dict__ and args.modal_gpu_type is not None:
        gpu_type = args.modal_gpu_type
    timeout_seconds = 86400
    if "modal_timeout_seconds" in args.__dict__ and args.modal_timeout_seconds is not None:
        timeout_seconds = args.modal_timeout_seconds
    poll_interval_seconds = 5
    if "modal_poll_interval_seconds" in args.__dict__ and args.modal_poll_interval_seconds is not None:
        poll_interval_seconds = args.modal_poll_interval_seconds
    log_tail_chars = 5000
    if "modal_log_tail_chars" in args.__dict__ and args.modal_log_tail_chars is not None:
        log_tail_chars = args.modal_log_tail_chars
    max_stale_heartbeat_seconds = 600
    if "modal_max_stale_heartbeat_seconds" in args.__dict__ and args.modal_max_stale_heartbeat_seconds is not None:
        max_stale_heartbeat_seconds = args.modal_max_stale_heartbeat_seconds
    artifacts_root = "modal_artifacts"
    if "modal_artifacts_dir" in args.__dict__ and args.modal_artifacts_dir is not None:
        artifacts_root = args.modal_artifacts_dir

    if args.rebuild_modal:
        print("Rebuilding Modal backend due to --rebuild_modal ...")
        _deploy_modal_backend(args)

    config = _build_modal_config_from_args(args)

    submit_fn = modal.Function.from_name(app_name, "submit_protify_job")
    status_fn = modal.Function.from_name(app_name, "get_job_status")
    log_delta_fn = modal.Function.from_name(app_name, "get_job_log_delta")
    results_fn = modal.Function.from_name(app_name, "get_results")
    delete_embeddings_fn = modal.Function.from_name(app_name, "delete_modal_embeddings")

    if args.delete_modal_embeddings:
        print("Deleting Modal embedding cache due to --delete_modal_embeddings ...")
        try:
            delete_embeddings_payload = delete_embeddings_fn.remote()
        except Exception:
            print("Modal embedding delete failed before app/function lookup succeeded; attempting deploy then retry...")
            _deploy_modal_backend(args)
            submit_fn = modal.Function.from_name(app_name, "submit_protify_job")
            status_fn = modal.Function.from_name(app_name, "get_job_status")
            log_delta_fn = modal.Function.from_name(app_name, "get_job_log_delta")
            results_fn = modal.Function.from_name(app_name, "get_results")
            delete_embeddings_fn = modal.Function.from_name(app_name, "delete_modal_embeddings")
            delete_embeddings_payload = delete_embeddings_fn.remote()
        if isinstance(delete_embeddings_payload, dict) and "message" in delete_embeddings_payload:
            print(delete_embeddings_payload["message"])

    has_dataset_run = len(args.data_names) > 0 or len(args.data_dirs) > 0
    if not has_dataset_run and not args.proteingym:
        return 0

    try:
        submit_result = submit_fn.remote(
            config=config,
            gpu_type=gpu_type,
            hf_token=args.hf_token,
            wandb_api_key=args.wandb_api_key,
            synthyra_api_key=args.synthyra_api_key,
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        print("Modal submit failed before app/function lookup succeeded; attempting deploy then retry...")
        _deploy_modal_backend(args)
        submit_fn = modal.Function.from_name(app_name, "submit_protify_job")
        status_fn = modal.Function.from_name(app_name, "get_job_status")
        log_delta_fn = modal.Function.from_name(app_name, "get_job_log_delta")
        results_fn = modal.Function.from_name(app_name, "get_results")
        submit_result = submit_fn.remote(
            config=config,
            gpu_type=gpu_type,
            hf_token=args.hf_token,
            wandb_api_key=args.wandb_api_key,
            synthyra_api_key=args.synthyra_api_key,
            timeout_seconds=timeout_seconds,
        )

    assert isinstance(submit_result, dict), "Modal submit response is not a dictionary."
    assert "job_id" in submit_result, "Modal submit response missing job_id."
    job_id = submit_result["job_id"]
    function_call_id = submit_result["function_call_id"] if "function_call_id" in submit_result else None
    print(f"Modal job submitted: {job_id}")
    if function_call_id is not None:
        print(f"Modal function call id: {function_call_id}")

    terminal_states = {"SUCCESS", "FAILED", "TERMINATED", "TIMEOUT"}
    final_status_payload = None
    poll_start_time = time.time()
    max_poll_seconds = int(timeout_seconds) + 900
    status_print_interval_seconds = 15
    last_status_print_time = 0.0
    last_status_line = ""
    missing_status_count = 0
    log_offset = 0
    function_call = None
    if function_call_id is not None:
        function_call = modal.FunctionCall.from_id(function_call_id)

    def _emit_remote_logs():
        nonlocal log_offset
        delta_payload = log_delta_fn.remote(job_id=job_id, offset=log_offset, max_chars=log_tail_chars)
        if isinstance(delta_payload, dict):
            if "next_offset" in delta_payload and isinstance(delta_payload["next_offset"], int):
                log_offset = delta_payload["next_offset"]
            if "chunk" in delta_payload and delta_payload["chunk"]:
                sys.stdout.write(delta_payload["chunk"])
                sys.stdout.flush()

    while True:
        _emit_remote_logs()

        status_payload = status_fn.remote(job_id=job_id)
        assert isinstance(status_payload, dict), "Modal status response is not a dictionary."
        if "success" in status_payload and status_payload["success"]:
            missing_status_count = 0
            status_value = status_payload["status"] if "status" in status_payload else "UNKNOWN"
            phase_value = status_payload["phase"] if "phase" in status_payload else "N/A"
            heartbeat_age = status_payload["heartbeat_age_seconds"] if "heartbeat_age_seconds" in status_payload else None
            heartbeat_text = "N/A" if heartbeat_age is None else f"{heartbeat_age:.1f}s"
            status_line = f"[Modal] status={status_value} phase={phase_value} heartbeat_age={heartbeat_text}"
            if status_value in terminal_states:
                final_status_payload = dict(status_payload)
                break
        else:
            missing_status_count += 1
            status_line = "[Modal] state=queued_or_initializing"
            if missing_status_count % 6 == 0 and "error" in status_payload and status_payload["error"]:
                status_line = f"[Modal] state=queued_or_initializing detail={status_payload['error']}"

        now = time.time()
        if status_line != last_status_line or (now - last_status_print_time) >= status_print_interval_seconds:
            print(status_line)
            last_status_line = status_line
            last_status_print_time = now

        if function_call is not None:
            try:
                remote_result = function_call.get(timeout=0)
                final_status_payload = _coerce_modal_terminal_payload(remote_result)
                if "phase" not in final_status_payload and "phase" in status_payload:
                    final_status_payload["phase"] = status_payload["phase"]
                break
            except TimeoutError:
                pass
            except Exception as error:
                final_status_payload = {"status": "FAILED", "error": f"Function call failed: {error}"}
                break

        elapsed_seconds = now - poll_start_time
        if elapsed_seconds > max_poll_seconds:
            final_status_payload = {
                "status": "TIMEOUT",
                "phase": "poll_timeout",
                "error": f"Polling exceeded timeout window ({max_poll_seconds} seconds).",
            }
            break

        if "success" in status_payload and status_payload["success"] and "heartbeat_age_seconds" in status_payload:
            heartbeat_age = status_payload["heartbeat_age_seconds"]
            if heartbeat_age is not None and heartbeat_age > max_stale_heartbeat_seconds and function_call is None:
                final_status_payload = {
                    "status": "FAILED",
                    "phase": "stale_heartbeat",
                    "error": f"Heartbeat stale for {heartbeat_age:.1f}s with no function_call_id available.",
                }
                break
        time.sleep(max(1, int(poll_interval_seconds)))

    final_delta_payload = log_delta_fn.remote(job_id=job_id, offset=log_offset, max_chars=log_tail_chars * 8)
    if isinstance(final_delta_payload, dict):
        if "chunk" in final_delta_payload and final_delta_payload["chunk"]:
            sys.stdout.write(final_delta_payload["chunk"])
            sys.stdout.flush()

    try:
        results_payload = results_fn.remote(job_id=job_id)
    except Exception as error:
        results_payload = {"success": False, "error": str(error)}
    if isinstance(results_payload, dict) and "success" in results_payload and results_payload["success"]:
        artifacts_dir = _save_modal_artifacts(results_payload, artifacts_root, job_id)
        print(f"Modal artifacts saved to {artifacts_dir}")

    if final_status_payload is None:
        final_status_payload = {"status": "FAILED", "error": "No terminal status was resolved."}

    final_status = final_status_payload["status"] if "status" in final_status_payload else "FAILED"
    if final_status != "SUCCESS":
        if "error" in final_status_payload and final_status_payload["error"]:
            print(f"Modal job failed: {final_status_payload['error']}")
        return 1
    return 0
