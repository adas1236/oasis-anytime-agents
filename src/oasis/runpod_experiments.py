"""Plan, launch, measure, and execute reproducible mock evaluations on Runpod Pods.

Planning and API mutation are deliberately separate.  ``plan`` is local and
read-only apart from its JSON output; ``launch``, ``stop``, and ``terminate``
only mutate Runpod state when their explicit ``--execute`` flag is present.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import tomllib
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Record this before importing the evaluator and its geospatial dependencies so
# the cost report attributes Python/module initialization to container startup.
_WORKER_PROCESS_STARTED_AT = datetime.now(UTC) if os.environ.get("OASIS_JOB_ID") else None

from oasis.mock_experiments import (  # noqa: E402 - preserve worker-start timing
    DATASET_FILES,
    DatasetKind,
    _parse_time_budgets,
    _parse_token_budgets,
    load_dataset,
    selection_digest,
)

RUNPOD_API_BASE = "https://api.runpod.io/v2"
_SECRET_REFERENCE = re.compile(r"\{\{\s*RUNPOD_SECRET_[A-Za-z0-9_-]+\s*\}\}")
_SENSITIVE_ENV_NAME = re.compile(r"(?:TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL)", re.I)


@dataclass(frozen=True, slots=True)
class PlanOverrides:
    rows: int | None = None
    seed: int | None = None
    shards: int | None = None
    time_budgets: str | None = None
    token_budgets: str | None = None
    gpu_types: tuple[str, ...] | None = None
    gpu_count: int | None = None
    model_type: str | None = None
    probe_only: bool | None = None


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "experiment"


def _section(document: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"TOML section [{name}] must be a table")
    return value


def _strings(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty TOML array")
    if any(not isinstance(item, (str, int, float)) for item in value):
        raise ValueError(f"{name} values must be strings or numbers")
    return [str(item) for item in value]


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return int(value)


def _canonical_time_budgets(values: Sequence[str]) -> list[str]:
    parsed = _parse_time_budgets(",".join(values))
    return ["unlimited" if value is None else f"{value:g}s" for value in parsed]


def _canonical_token_budgets(values: Sequence[str]) -> list[str]:
    parsed = _parse_token_budgets(",".join(values))
    return ["unlimited" if value is None else str(value) for value in parsed]


def _split_sizes(total: int, parts: int) -> list[int]:
    if parts > total:
        raise ValueError("shards_per_condition cannot exceed rows_per_dataset")
    quotient, remainder = divmod(total, parts)
    return [quotient + (1 if index < remainder else 0) for index in range(parts)]


def _safe_environment(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("[runpod.env] must be a TOML table")
    result: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, (str, int, float, bool)):
            raise ValueError("[runpod.env] keys and scalar values must be strings")
        name = raw_name.strip()
        item = str(raw_value)
        if name.startswith("OASIS_") or name == "CUDA_VISIBLE_DEVICES":
            raise ValueError(f"[runpod.env] cannot override reserved worker variable {name}")
        if _SENSITIVE_ENV_NAME.search(name) and _SECRET_REFERENCE.fullmatch(item) is None:
            raise ValueError(
                f"Refusing to put a literal secret in the plan for {name}; use a "
                "{{ RUNPOD_SECRET_name }} reference"
            )
        result[name] = item
    return result


def _selection_plan(
    *, dataset: str, data_root: Path, rows: int, seed: int, shards: int
) -> dict[str, Any]:
    cases = load_dataset(DatasetKind(dataset), data_root / DATASET_FILES[dataset])
    random.Random(seed).shuffle(cases)
    if rows > len(cases):
        raise ValueError(f"rows_per_dataset={rows} exceeds the {len(cases)} records in {dataset}")
    chosen = cases[:rows]
    chosen_ids = [case.record_id for case in chosen]
    shard_specs: list[dict[str, Any]] = []
    offset = 0
    for index, size in enumerate(_split_sizes(rows, shards)):
        record_ids = chosen_ids[offset : offset + size]
        shard_specs.append(
            {
                "index": index,
                "start": offset,
                "limit": size,
                "record_ids": record_ids,
                "selection_digest": selection_digest(record_ids),
            }
        )
        offset += size
    return {
        "dataset": dataset,
        "rows": rows,
        "seed": seed,
        "record_ids": chosen_ids,
        "selection_digest": selection_digest(chosen_ids),
        "shards": shard_specs,
    }


def _job_environment(
    *,
    plan_id: str,
    job_id: str,
    experiment: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    dataset: str,
    time_budget: str,
    token_budget: str,
    seed: int,
    shard: Mapping[str, Any],
    gpu_count: int,
    image: str,
) -> dict[str, str]:
    values = {
        "OASIS_PLAN_ID": plan_id,
        "OASIS_JOB_ID": job_id,
        "OASIS_DATASET": dataset,
        "OASIS_TIME_BUDGET": time_budget,
        "OASIS_TOKEN_BUDGET": token_budget,
        "OASIS_SEED": str(seed),
        "OASIS_START": str(shard["start"]),
        "OASIS_LIMIT": str(shard["limit"]),
        "OASIS_SELECTION_DIGEST": str(shard["selection_digest"]),
        "OASIS_MODEL_TYPE": str(experiment["model_type"]),
        "OASIS_MODEL_PROFILE": str(experiment["profile"]),
        "OASIS_PROBE_ONLY": "1" if experiment["probe_only"] else "0",
        "OASIS_IMAGE": image,
        "OASIS_DTYPE": str(experiment["dtype"]),
        "OASIS_QUANTIZATION": str(experiment["quantization"]),
        "OASIS_ATTENTION_BACKEND": str(experiment["attention_backend"]),
        "OASIS_THINKING": "1" if experiment["thinking"] else "0",
        "OASIS_MAX_GENERATED_TOKENS": str(experiment["max_generated_tokens"]),
        "OASIS_MAX_TOOL_ROUNDS": str(experiment["max_tool_rounds"]),
        "OASIS_MAX_TOOL_CALLS": str(experiment["max_tool_calls"]),
        "OASIS_MODEL_CALL_TIMEOUT": str(experiment["model_call_timeout"]),
        "OASIS_OSRM_CACHE_ONLY": "1" if experiment["osrm_cache_only"] else "0",
        "OASIS_OSRM_CACHE": str(experiment["osrm_cache"]),
        "OASIS_OSRM_ENDPOINT": str(experiment["osrm_endpoint"]),
        "OASIS_TSP_TOLERANCE_KM": str(experiment["tsp_tolerance_km"]),
        "OASIS_EXPECTED_GPU_COUNT": str(gpu_count),
        # Runpod keeps a Pod in its desired RUNNING state by restarting a
        # container that exits. Hold after final persistence so a monitor can
        # stop the Pod without rerunning the evaluation.
        "OASIS_HOLD_AFTER_COMPLETION": "1",
        "OASIS_RESULTS_ROOT": str(artifacts["local_root"]),
        "OASIS_RESULTS_S3_URI": str(artifacts["s3_uri"]),
        "OASIS_UPLOAD_INTERVAL_SECONDS": str(artifacts["upload_interval_seconds"]),
    }
    if experiment.get("model"):
        values["OASIS_MODEL_ID"] = str(experiment["model"])
    if experiment.get("revision"):
        values["OASIS_MODEL_REVISION"] = str(experiment["revision"])
    return values


def _pod_payload(
    *,
    pod_name: str,
    runpod: Mapping[str, Any],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    merged_environment = dict(environment) | dict(runpod["env"])
    if len(merged_environment) > 50:
        raise ValueError("Runpod permits at most 50 environment variables per Pod")
    payload: dict[str, Any] = {
        "name": pod_name[:191],
        "image": runpod["image"],
        "cloud": runpod["cloud_type"],
        "gpu": {
            "id": runpod["gpu_types"][0],
            "count": runpod["gpu_count"],
            "minVcpuCountPerGpu": runpod["min_vcpu_per_gpu"],
            "minRamPerGpu": runpod["min_ram_per_gpu_gb"],
        },
        "disk": runpod["container_disk_gb"],
        "globalNetworking": False,
        "ports": [],
        "env": merged_environment,
    }
    if runpod["network_volume_id"]:
        payload["mounts"] = {
            "network": [
                {
                    "volumeId": runpod["network_volume_id"],
                    "path": runpod["volume_mount_path"],
                }
            ]
        }
    elif runpod["volume_disk_gb"]:
        payload["mounts"] = {
            "persistent": {
                "size": runpod["volume_disk_gb"],
                "path": runpod["volume_mount_path"],
            }
        }
    if runpod["data_center_ids"]:
        payload["dataCenterIds"] = runpod["data_center_ids"]
    if runpod["allowed_cuda_versions"]:
        payload["gpu"]["allowedCudaVersions"] = runpod["allowed_cuda_versions"]
    if runpod["container_registry_auth_id"]:
        payload["registry"] = runpod["container_registry_auth_id"]
    return payload


def build_plan(config_path: Path, overrides: PlanOverrides | None = None) -> dict[str, Any]:
    """Build a deterministic plan without contacting Runpod."""

    overrides = overrides or PlanOverrides()

    try:
        document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Configuration does not exist: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {config_path}: {exc}") from exc
    experiment_raw = _section(document, "experiment")
    runpod_raw = _section(document, "runpod")
    artifacts_raw = _section(document, "artifacts")

    name = str(experiment_raw.get("name", "oasis-mock-eval"))
    data_root = Path(str(experiment_raw.get("data_root", "data")))
    if not data_root.is_absolute():
        data_root = (config_path.parent / data_root).resolve()
    dataset_values = _strings(
        experiment_raw.get("datasets", list(DATASET_FILES)), "experiment.datasets"
    )
    datasets = list(dict.fromkeys(dataset_values))
    if any(dataset not in DATASET_FILES for dataset in datasets):
        raise ValueError(f"datasets must be chosen from {sorted(DATASET_FILES)}")
    rows = _positive_int(
        overrides.rows
        if overrides.rows is not None
        else experiment_raw.get("rows_per_dataset", 500),
        "rows_per_dataset",
    )
    seed = _positive_int(
        overrides.seed if overrides.seed is not None else experiment_raw.get("seed", 42),
        "seed",
        allow_zero=True,
    )
    shards = _positive_int(
        overrides.shards
        if overrides.shards is not None
        else experiment_raw.get("shards_per_condition", 1),
        "shards_per_condition",
    )
    raw_times = (
        overrides.time_budgets.split(",")
        if overrides.time_budgets is not None
        else _strings(experiment_raw.get("time_budgets", ["unlimited"]), "time_budgets")
    )
    raw_tokens = (
        overrides.token_budgets.split(",")
        if overrides.token_budgets is not None
        else _strings(experiment_raw.get("token_budgets", ["unlimited"]), "token_budgets")
    )
    time_budgets = _canonical_time_budgets(raw_times)
    token_budgets = _canonical_token_budgets(raw_tokens)
    model_type = overrides.model_type or str(experiment_raw.get("model_type", "transformers"))
    if model_type not in {"fake", "transformers"}:
        raise ValueError("model_type must be fake or transformers")

    experiment: dict[str, Any] = {
        "name": name,
        "data_root": str(data_root),
        "datasets": datasets,
        "rows_per_dataset": rows,
        "seed": seed,
        "shards_per_condition": shards,
        "time_budgets": time_budgets,
        "token_budgets": token_budgets,
        "model_type": model_type,
        "profile": str(experiment_raw.get("profile", "gemma4_e2b_it")),
        "probe_only": (
            overrides.probe_only
            if overrides.probe_only is not None
            else bool(experiment_raw.get("probe_only", False))
        ),
        "model": experiment_raw.get("model"),
        "revision": experiment_raw.get("revision"),
        "dtype": str(experiment_raw.get("dtype", "bfloat16")),
        "quantization": str(experiment_raw.get("quantization", "none")),
        "attention_backend": str(experiment_raw.get("attention_backend", "sdpa")),
        "thinking": bool(experiment_raw.get("thinking", True)),
        "max_generated_tokens": _positive_int(
            experiment_raw.get("max_generated_tokens", 768), "max_generated_tokens"
        ),
        "max_tool_rounds": _positive_int(
            experiment_raw.get("max_tool_rounds", 4), "max_tool_rounds"
        ),
        "max_tool_calls": str(experiment_raw.get("max_tool_calls", "unlimited")),
        "model_call_timeout": str(experiment_raw.get("model_call_timeout", "unlimited")),
        "osrm_cache_only": bool(experiment_raw.get("osrm_cache_only", False)),
        "osrm_cache": str(experiment_raw.get("osrm_cache", "/opt/oasis/osrm-cache")),
        "osrm_endpoint": str(
            experiment_raw.get("osrm_endpoint", "https://router.project-osrm.org")
        ),
        "tsp_tolerance_km": float(experiment_raw.get("tsp_tolerance_km", 1.0)),
    }
    image = str(runpod_raw.get("image", "")).strip()
    if not image:
        raise ValueError("runpod.image is required")
    configured_gpu_types = _strings(
        runpod_raw.get("gpu_types", ["NVIDIA GeForce RTX 5090"]), "runpod.gpu_types"
    )
    gpu_types = list(overrides.gpu_types or tuple(configured_gpu_types))
    if len(gpu_types) != 1:
        raise ValueError(
            "Runpod v2 plans require exactly one gpu_type; create separate plans "
            "to compare or fall back across GPU types"
        )
    gpu_count = _positive_int(
        overrides.gpu_count if overrides.gpu_count is not None else runpod_raw.get("gpu_count", 1),
        "runpod.gpu_count",
    )
    cloud_type = str(runpod_raw.get("cloud_type", "SECURE")).upper()
    if cloud_type not in {"SECURE", "COMMUNITY"}:
        raise ValueError("runpod.cloud_type must be SECURE or COMMUNITY")
    runpod: dict[str, Any] = {
        "image": image,
        "gpu_types": gpu_types,
        "gpu_count": gpu_count,
        "cloud_type": cloud_type,
        "container_disk_gb": _positive_int(
            runpod_raw.get("container_disk_gb", 60), "runpod.container_disk_gb"
        ),
        "volume_disk_gb": _positive_int(
            runpod_raw.get("volume_disk_gb", 40),
            "runpod.volume_disk_gb",
            allow_zero=True,
        ),
        "volume_mount_path": str(runpod_raw.get("volume_mount_path", "/workspace")),
        "network_volume_id": str(runpod_raw.get("network_volume_id", "")),
        "container_registry_auth_id": str(runpod_raw.get("container_registry_auth_id", "")),
        "interruptible": bool(runpod_raw.get("interruptible", False)),
        "data_center_ids": _strings(runpod_raw["data_center_ids"], "data_center_ids")
        if runpod_raw.get("data_center_ids")
        else [],
        "allowed_cuda_versions": _strings(
            runpod_raw["allowed_cuda_versions"], "allowed_cuda_versions"
        )
        if runpod_raw.get("allowed_cuda_versions")
        else [],
        "min_vcpu_per_gpu": _positive_int(
            runpod_raw.get("min_vcpu_per_gpu", 4), "runpod.min_vcpu_per_gpu"
        ),
        "min_ram_per_gpu_gb": _positive_int(
            runpod_raw.get("min_ram_per_gpu_gb", 16), "runpod.min_ram_per_gpu_gb"
        ),
        "env": _safe_environment(runpod_raw.get("env")),
    }
    if runpod["interruptible"]:
        raise ValueError("runpod.interruptible is not supported by the Runpod v2 Pod API")
    if 0 < runpod["volume_disk_gb"] < 10:
        raise ValueError("runpod.volume_disk_gb must be zero or at least 10 GB")
    artifacts: dict[str, Any] = {
        "local_root": str(artifacts_raw.get("local_root", "/workspace/oasis-results")),
        "s3_uri": str(artifacts_raw.get("s3_uri", "")),
        "upload_interval_seconds": _positive_int(
            artifacts_raw.get("upload_interval_seconds", 60),
            "artifacts.upload_interval_seconds",
        ),
    }
    if artifacts["s3_uri"]:
        _s3_location(artifacts["s3_uri"])
    if not artifacts["s3_uri"] and not runpod["network_volume_id"] and not runpod["volume_disk_gb"]:
        raise ValueError("Configure S3, a network volume, or a nonzero Pod volume for results")

    selections = {
        dataset: _selection_plan(
            dataset=dataset,
            data_root=data_root,
            rows=rows,
            seed=seed,
            shards=shards,
        )
        for dataset in datasets
    }
    identity = {
        "experiment": experiment,
        "runpod": runpod,
        "artifacts": artifacts,
        "selection_digests": {
            dataset: selection["selection_digest"] for dataset, selection in selections.items()
        },
    }
    plan_id = f"{_slug(name)}-{_json_hash(identity)[:10]}"
    jobs: list[dict[str, Any]] = []
    for dataset, time_budget, token_budget in itertools.product(
        datasets, time_budgets, token_budgets
    ):
        for shard in selections[dataset]["shards"]:
            time_slug = _slug(time_budget)
            token_slug = _slug(token_budget)
            job_id = (
                f"{dataset}__time-{time_slug}__tokens-{token_slug}__shard-{int(shard['index']):02d}"
            )
            environment = _job_environment(
                plan_id=plan_id,
                job_id=job_id,
                experiment=experiment,
                artifacts=artifacts,
                dataset=dataset,
                time_budget=time_budget,
                token_budget=token_budget,
                seed=seed,
                shard=shard,
                gpu_count=gpu_count,
                image=image,
            )
            pod_name = f"{plan_id}-{job_id}"
            jobs.append(
                {
                    "job_id": job_id,
                    "dataset": dataset,
                    "time_budget": time_budget,
                    "token_budget": token_budget,
                    "shard_index": shard["index"],
                    "start": shard["start"],
                    "limit": shard["limit"],
                    "selection_digest": shard["selection_digest"],
                    "record_ids": shard["record_ids"],
                    "pod_payload": _pod_payload(
                        pod_name=pod_name,
                        runpod=runpod,
                        environment=environment,
                    ),
                }
            )
    plan: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "plan_id": plan_id,
        "experiment": experiment,
        "runpod": runpod,
        "artifacts": artifacts,
        "selections": selections,
        "jobs": jobs,
    }
    plan["plan_fingerprint"] = _plan_fingerprint(plan)
    return plan


def _plan_fingerprint(plan: Mapping[str, Any]) -> str:
    stable = {
        key: value for key, value in plan.items() if key not in {"created_at", "plan_fingerprint"}
    }
    return _json_hash(stable)


def _placeholder_paths(value: Any, path: str = "") -> list[str]:
    matches: list[str] = []
    if isinstance(value, str) and "REPLACE_" in value.upper():
        matches.append(path or "<root>")
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            matches.extend(_placeholder_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            matches.extend(_placeholder_paths(item, f"{path}[{index}]"))
    return matches


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def load_plan(path: Path) -> dict[str, Any]:
    plan = _load_json_object(path, "plan")
    if plan.get("plan_fingerprint") != _plan_fingerprint(plan):
        raise ValueError(f"Plan fingerprint does not match its contents: {path}")
    if not isinstance(plan.get("jobs"), list):
        raise ValueError(f"Plan has no jobs array: {path}")
    return plan


class RunpodApi:
    def __init__(self, api_key: str, base_url: str = RUNPOD_API_BASE) -> None:
        if not api_key:
            raise ValueError("RUNPOD_API_KEY is required for --execute")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2_000]
            raise RuntimeError(f"Runpod API {method} {path} failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Runpod API {method} {path} failed: {exc.reason}") from exc
        return json.loads(content) if content else None


def _selected_jobs(
    plan: Mapping[str, Any], requested_ids: Sequence[str], maximum: int | None
) -> list[dict[str, Any]]:
    jobs = plan["jobs"]
    if not isinstance(jobs, list) or any(not isinstance(job, dict) for job in jobs):
        raise ValueError("Plan jobs are malformed")
    known = {str(job["job_id"]) for job in jobs}
    unknown = set(requested_ids) - known
    if unknown:
        raise ValueError(f"Unknown job IDs: {sorted(unknown)}")
    selected = [job for job in jobs if not requested_ids or job["job_id"] in requested_ids]
    return selected if maximum is None else selected[:maximum]


def _state_path(plan_path: Path, requested: Path | None) -> Path:
    return requested or plan_path.with_suffix(".state.json")


def _new_or_existing_state(path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        state = _load_json_object(path, "launch state")
        if state.get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise ValueError("Launch state belongs to a different plan")
        if not isinstance(state.get("pods"), dict):
            raise ValueError("Launch state pods value is malformed")
        return state
    return {
        "plan_id": plan["plan_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "pods": {},
    }


def launch_plan(
    *,
    plan_path: Path,
    state_path: Path | None,
    execute: bool,
    requested_ids: Sequence[str] = (),
    maximum: int | None = None,
    print_payloads: bool = False,
    api_base: str = RUNPOD_API_BASE,
) -> int:
    if maximum is not None and maximum < 1:
        raise ValueError("maximum jobs must be positive")
    plan = load_plan(plan_path)
    path = _state_path(plan_path, state_path)
    state = _new_or_existing_state(path, plan)
    jobs = _selected_jobs(plan, requested_ids, maximum)
    pending = [job for job in jobs if job["job_id"] not in state["pods"]]
    print(
        f"Plan {plan['plan_id']}: {len(jobs)} selected, {len(pending)} not yet launched; "
        f"GPU={plan['runpod']['gpu_count']} x {plan['runpod']['gpu_types']}"
    )
    if print_payloads:
        for job in pending:
            print(json.dumps(job["pod_payload"], indent=2, ensure_ascii=False))
    if not execute:
        print("Dry run only: no Pods were created. Add --execute to launch them.")
        return 0
    placeholders = _placeholder_paths({"runpod": plan["runpod"], "artifacts": plan["artifacts"]})
    if placeholders:
        raise ValueError(
            "Replace required configuration placeholders before launching: "
            + ", ".join(placeholders)
        )
    if not pending:
        print(f"Nothing to launch; existing state is {path}")
        return 0
    client = RunpodApi(os.environ.get("RUNPOD_API_KEY", ""), api_base)
    for position, job in enumerate(pending, start=1):
        print(f"[{position}/{len(pending)}] launching {job['job_id']}", file=sys.stderr)
        created = client.request("POST", "/pods", job["pod_payload"])
        if not isinstance(created, dict) or not isinstance(created.get("id"), str):
            raise RuntimeError(f"Runpod returned no Pod ID for {job['job_id']}")
        state["pods"][job["job_id"]] = {
            "pod_id": created["id"],
            "name": created.get("name", job["pod_payload"]["name"]),
            "status": created.get("status"),
            "cost_per_hour": created.get("cost"),
            "created_at": created.get("createdAt"),
            "started_at": created.get("startedAt"),
            "launched_at": datetime.now(UTC).isoformat(),
            "stopped_at": None,
            "terminated_at": None,
        }
        state["updated_at"] = datetime.now(UTC).isoformat()
        _write_json(path, state)
    print(f"Launch state: {path}")
    return 0


def show_status(*, state_path: Path, api_base: str = RUNPOD_API_BASE) -> int:
    state = _load_json_object(state_path, "launch state")
    pods = state.get("pods")
    if not isinstance(pods, dict):
        raise ValueError("Launch state pods value is malformed")
    client = RunpodApi(os.environ.get("RUNPOD_API_KEY", ""), api_base)
    rows: list[dict[str, Any]] = []
    for job_id, item in pods.items():
        if not isinstance(item, dict) or not isinstance(item.get("pod_id"), str):
            raise ValueError(f"Malformed Pod state for {job_id}")
        if item.get("terminated_at"):
            rows.append({"job_id": job_id, "pod_id": item["pod_id"], "status": "TERMINATED"})
            continue
        response = client.request("GET", f"/pods/{urllib.parse.quote(item['pod_id'], safe='')}")
        pod = response if isinstance(response, dict) else {}
        cost_per_hour = item.get("cost_per_hour") or pod.get("cost")
        billing_start = item.get("created_at") or item.get("launched_at")
        billing_end = item.get("stopped_at") or item.get("terminated_at")
        rows.append(
            {
                "job_id": job_id,
                "pod_id": item["pod_id"],
                "status": pod.get("status", "NOT_FOUND"),
                "gpu": pod.get("gpu"),
                "cost_per_hour": cost_per_hour,
                "created_at": pod.get("createdAt") or billing_start,
                "started_at": pod.get("startedAt") or item.get("started_at"),
                "runtime": pod.get("runtime"),
                "estimated_compute_cost_usd": _estimated_cost(
                    billing_start, billing_end, cost_per_hour
                ),
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _elapsed_seconds(start: Any, end: Any = None) -> float | None:
    started_at = _parse_timestamp(start)
    if started_at is None:
        return None
    finished_at = _parse_timestamp(end) if end is not None else datetime.now(UTC)
    if finished_at is None:
        return None
    return max(0.0, (finished_at - started_at).total_seconds())


def _estimated_cost(start: Any, end: Any, cost_per_hour: Any) -> float | None:
    if isinstance(cost_per_hour, bool) or not isinstance(cost_per_hour, (int, float)):
        return None
    seconds = _elapsed_seconds(start, end)
    return None if seconds is None else round(seconds * float(cost_per_hour) / 3600, 6)


def stop_pods(*, state_path: Path, execute: bool, api_base: str = RUNPOD_API_BASE) -> int:
    """Stop compute while retaining each Pod and its persistent disk."""

    state = _load_json_object(state_path, "launch state")
    pods = state.get("pods")
    if not isinstance(pods, dict):
        raise ValueError("Launch state pods value is malformed")
    pending = [
        (job_id, item)
        for job_id, item in pods.items()
        if isinstance(item, dict) and not item.get("stopped_at") and not item.get("terminated_at")
    ]
    print(f"{len(pending)} Pods in {state_path} are eligible to stop.")
    if not execute:
        print("Dry run only: no Pods were stopped. Add --execute to release their compute.")
        return 0
    if not pending:
        return 0
    client = RunpodApi(os.environ.get("RUNPOD_API_KEY", ""), api_base)
    for position, (job_id, item) in enumerate(pending, start=1):
        pod_id = item.get("pod_id")
        if not isinstance(pod_id, str):
            raise ValueError(f"Malformed Pod ID for {job_id}")
        encoded_id = urllib.parse.quote(pod_id, safe="")
        current = client.request("GET", f"/pods/{encoded_id}")
        if not isinstance(current, dict):
            raise RuntimeError(f"Runpod returned a malformed Pod for {job_id}")
        status = str(current.get("status", ""))
        if status not in {"EXITED", "ERROR"}:
            print(f"[{position}/{len(pending)}] stopping {job_id} ({pod_id})", file=sys.stderr)
            stopped = client.request("POST", f"/pods/{encoded_id}/action", {"action": "stop"})
            if not isinstance(stopped, dict):
                raise RuntimeError(f"Runpod returned no stopped Pod for {job_id}")
            status = str(stopped.get("status", "EXITED"))
        stopped_at = datetime.now(UTC).isoformat()
        item["stopped_at"] = stopped_at
        item["status"] = status
        item["estimated_compute_cost_usd"] = _estimated_cost(
            item.get("created_at") or item.get("launched_at"),
            stopped_at,
            item.get("cost_per_hour"),
        )
        state["updated_at"] = stopped_at
        _write_json(state_path, state)
    return 0


def cost_report(*, state_path: Path, results_root: Path) -> int:
    """Combine launch state and downloaded job statuses into a timing/cost report."""

    state = _load_json_object(state_path, "launch state")
    plan_id = state.get("plan_id")
    pods = state.get("pods")
    if not isinstance(plan_id, str) or not isinstance(pods, dict):
        raise ValueError("Launch state is malformed")
    plan_results = results_root if results_root.name == plan_id else results_root / plan_id
    rows: list[dict[str, Any]] = []
    for job_id, item in pods.items():
        if not isinstance(item, dict):
            raise ValueError(f"Malformed Pod state for {job_id}")
        status_path = plan_results / str(job_id) / "job-status.json"
        job_status = _load_json_object(status_path, "job status") if status_path.is_file() else {}
        created_at = item.get("created_at") or item.get("launched_at")
        container_started_at = job_status.get("container_started_at") or job_status.get(
            "started_at"
        )
        workload_started_at = job_status.get("started_at")
        finished_at = job_status.get("finished_at")
        stopped_at = item.get("stopped_at") or item.get("terminated_at")
        cost_per_hour = item.get("cost_per_hour")
        rows.append(
            {
                "job_id": job_id,
                "pod_id": item.get("pod_id"),
                "mode": job_status.get("mode"),
                "status": job_status.get("status", item.get("status", "missing_artifacts")),
                "image": job_status.get("image"),
                "created_at": created_at,
                "container_started_at": container_started_at,
                "workload_started_at": workload_started_at,
                "finished_at": finished_at,
                "stopped_at": stopped_at,
                "image_pull_and_container_start_seconds": _elapsed_seconds(
                    created_at, container_started_at
                ),
                "worker_setup_seconds": _elapsed_seconds(container_started_at, workload_started_at),
                "workload_seconds": job_status.get("elapsed_seconds"),
                "completion_to_stop_seconds": _elapsed_seconds(finished_at, stopped_at)
                if stopped_at
                else None,
                "billed_seconds_estimate": _elapsed_seconds(created_at, stopped_at),
                "cost_per_hour": cost_per_hour,
                "estimated_compute_cost_usd": _estimated_cost(
                    created_at, stopped_at, cost_per_hour
                ),
                "artifacts_found": bool(job_status),
            }
        )
    costs = [
        float(row["estimated_compute_cost_usd"])
        for row in rows
        if isinstance(row["estimated_compute_cost_usd"], (int, float))
    ]
    print(
        json.dumps(
            {
                "plan_id": plan_id,
                "jobs": rows,
                "summary": {
                    "job_count": len(rows),
                    "jobs_with_artifacts": sum(bool(row["artifacts_found"]) for row in rows),
                    "estimated_compute_cost_usd": round(sum(costs), 6),
                    "complete": all(row["stopped_at"] for row in rows),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def terminate_pods(*, state_path: Path, execute: bool, api_base: str = RUNPOD_API_BASE) -> int:
    state = _load_json_object(state_path, "launch state")
    pods = state.get("pods")
    if not isinstance(pods, dict):
        raise ValueError("Launch state pods value is malformed")
    pending = [
        (job_id, item)
        for job_id, item in pods.items()
        if isinstance(item, dict) and not item.get("terminated_at")
    ]
    print(f"{len(pending)} Pods in {state_path} are eligible for permanent deletion.")
    if not execute:
        print("Dry run only: no Pods were deleted. Add --execute after verifying artifacts.")
        return 0
    if not pending:
        return 0
    client = RunpodApi(os.environ.get("RUNPOD_API_KEY", ""), api_base)
    for position, (job_id, item) in enumerate(pending, start=1):
        pod_id = item.get("pod_id")
        if not isinstance(pod_id, str):
            raise ValueError(f"Malformed Pod ID for {job_id}")
        print(f"[{position}/{len(pending)}] deleting {job_id} ({pod_id})", file=sys.stderr)
        client.request("DELETE", f"/pods/{urllib.parse.quote(pod_id, safe='')}")
        item["terminated_at"] = datetime.now(UTC).isoformat()
        item["status"] = "TERMINATED"
        state["updated_at"] = datetime.now(UTC).isoformat()
        _write_json(state_path, state)
    return 0


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Container environment variable {name} is required")
    return value


def _runner_arguments(output: Path) -> list[str]:
    arguments = [
        sys.executable,
        "-m",
        "oasis.run_mock_experiment",
        "--model-type",
        _required_env("OASIS_MODEL_TYPE"),
        "--profile",
        _required_env("OASIS_MODEL_PROFILE"),
        "--dataset",
        _required_env("OASIS_DATASET"),
        "--gpus",
        "auto",
        "--dtype",
        _required_env("OASIS_DTYPE"),
        "--quantization",
        _required_env("OASIS_QUANTIZATION"),
        "--attention-backend",
        _required_env("OASIS_ATTENTION_BACKEND"),
        "--time-budgets",
        _required_env("OASIS_TIME_BUDGET"),
        "--token-budgets",
        _required_env("OASIS_TOKEN_BUDGET"),
        "--max-generated-tokens",
        _required_env("OASIS_MAX_GENERATED_TOKENS"),
        "--max-tool-rounds",
        _required_env("OASIS_MAX_TOOL_ROUNDS"),
        "--max-tool-calls",
        _required_env("OASIS_MAX_TOOL_CALLS"),
        "--model-call-timeout-seconds",
        _required_env("OASIS_MODEL_CALL_TIMEOUT"),
        "--start",
        _required_env("OASIS_START"),
        "--limit",
        _required_env("OASIS_LIMIT"),
        "--shuffle",
        "--seed",
        _required_env("OASIS_SEED"),
        "--expected-selection-digest",
        _required_env("OASIS_SELECTION_DIGEST"),
        "--osrm-endpoint",
        _required_env("OASIS_OSRM_ENDPOINT"),
        "--osrm-cache",
        _required_env("OASIS_OSRM_CACHE"),
        "--tsp-tolerance-km",
        _required_env("OASIS_TSP_TOLERANCE_KM"),
        "--output",
        str(output),
        "--resume",
    ]
    arguments.append("--thinking" if _required_env("OASIS_THINKING") == "1" else "--no-thinking")
    if os.environ.get("OASIS_OSRM_CACHE_ONLY") == "1":
        arguments.append("--osrm-cache-only")
    if model := os.environ.get("OASIS_MODEL_ID", "").strip():
        arguments.extend(("--model", model))
    if revision := os.environ.get("OASIS_MODEL_REVISION", "").strip():
        arguments.extend(("--revision", revision))
    return arguments


def _s3_location(uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("artifacts.s3_uri must be empty or an s3://bucket/prefix URI")
    return parsed.netloc, parsed.path.strip("/")


class ArtifactSync:
    def __init__(self, local_dir: Path, s3_uri: str) -> None:
        self.local_dir = local_dir
        self.s3_uri = s3_uri.strip()
        self.bucket = ""
        self.prefix = ""
        self.client: Any = None
        if self.s3_uri:
            self.bucket, self.prefix = _s3_location(self.s3_uri)
            try:
                boto3 = importlib.import_module("boto3")
            except ImportError as exc:
                raise RuntimeError("S3 artifact sync requires the Runpod dependency group") from exc
            endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
            self.client = boto3.client("s3", endpoint_url=endpoint)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _key(self, path: Path) -> str:
        relative = path.relative_to(self.local_dir).as_posix()
        return "/".join(part for part in (self.prefix, relative) if part)

    def restore(self) -> None:
        if not self.enabled:
            return
        for name in ("results.jsonl", "results.summary.json", "results.jsonl.interrupted"):
            destination = self.local_dir / name
            if destination.exists():
                continue
            temporary = destination.with_suffix(destination.suffix + ".download")
            try:
                self.client.download_file(self.bucket, self._key(destination), str(temporary))
            except self.client.exceptions.ClientError as exc:
                code = str(exc.response.get("Error", {}).get("Code", ""))
                if code not in {"404", "NoSuchKey", "NotFound"}:
                    raise
                continue
            temporary.replace(destination)

    def upload(self) -> None:
        if not self.enabled:
            return
        for path in self.local_dir.iterdir():
            if path.is_file() and not path.name.endswith((".tmp", ".download")):
                self.client.upload_file(str(path), self.bucket, self._key(path))


def _gpu_inventory() -> dict[str, Any]:
    torch = importlib.import_module("torch")
    devices = []
    for index in range(int(torch.cuda.device_count())):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return {
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(getattr(torch.version, "cuda", None)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": devices,
    }


def _read_log_tail(path: Path, *, max_bytes: int = 16_384) -> str:
    """Return a bounded, UTF-8-safe log tail for remote failure diagnostics."""

    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            return stream.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"Could not read {path.name}: {type(exc).__name__}: {exc}"


def _evaluation_summary(path: Path) -> dict[str, Any] | None:
    """Read the durable evaluator summary used to validate a successful exit."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _hold_for_controller() -> None:
    """Keep a finished Runpod container alive until the external monitor stops it."""

    if os.environ.get("OASIS_HOLD_AFTER_COMPLETION", "0").strip() != "1":
        return
    print(
        json.dumps(
            {
                "event": "oasis_worker_awaiting_stop",
                "pod_id": os.environ.get("RUNPOD_POD_ID"),
            }
        ),
        flush=True,
    )
    while True:
        signal.pause()


def worker_main() -> int:
    """Run one planned condition inside a Pod and continuously persist artifacts."""

    container_started_at = _WORKER_PROCESS_STARTED_AT or datetime.now(UTC)
    plan_id = _required_env("OASIS_PLAN_ID")
    job_id = _required_env("OASIS_JOB_ID")
    probe_only = _required_env("OASIS_PROBE_ONLY") == "1"
    mode = "probe" if probe_only else "evaluation"
    expected_gpus = int(_required_env("OASIS_EXPECTED_GPU_COUNT"))
    root = Path(_required_env("OASIS_RESULTS_ROOT"))
    local_dir = root / plan_id / job_id
    local_dir.mkdir(parents=True, exist_ok=True)
    output = local_dir / "results.jsonl"
    status_path = local_dir / "job-status.json"
    stdout_path = local_dir / "runner.out.log"
    stderr_path = local_dir / "runner.err.log"
    configured_s3_uri = os.environ.get("OASIS_RESULTS_S3_URI", "").rstrip("/")
    job_s3_uri = f"{configured_s3_uri}/{plan_id}/{job_id}" if configured_s3_uri else ""
    sync = ArtifactSync(local_dir, job_s3_uri)
    sync.restore()
    hardware = _gpu_inventory()
    visible_gpus = len(hardware["devices"])
    if visible_gpus != expected_gpus:
        raise RuntimeError(
            f"Runpod allocated {visible_gpus} visible GPUs; plan requires {expected_gpus}"
        )
    started_at = datetime.now(UTC)
    common_status = {
        "mode": mode,
        "plan_id": plan_id,
        "job_id": job_id,
        "image": os.environ.get("OASIS_IMAGE"),
        "container_started_at": container_started_at.isoformat(),
        "started_at": started_at.isoformat(),
        "startup_elapsed_seconds": (started_at - container_started_at).total_seconds(),
        "visible_gpu_count": visible_gpus,
        "hardware": hardware,
        "selection_digest": os.environ["OASIS_SELECTION_DIGEST"],
    }
    _write_json(
        status_path,
        {
            "status": "running",
            **common_status,
        },
    )
    print(
        json.dumps(
            {
                "event": "oasis_worker_started",
                "mode": mode,
                "plan_id": plan_id,
                "job_id": job_id,
                "visible_gpu_count": visible_gpus,
            }
        ),
        flush=True,
    )
    if probe_only:
        finished_at = datetime.now(UTC)
        status = {
            "status": "complete",
            **common_status,
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": (finished_at - started_at).total_seconds(),
            "worker_elapsed_seconds": (finished_at - container_started_at).total_seconds(),
            "runner_exit_code": 0,
        }
        _write_json(status_path, status)
        try:
            sync.upload()
        except Exception as exc:
            probe_upload_error = f"{type(exc).__name__}: {exc}"
            status["status"] = "artifact_upload_failed"
            status["artifact_upload_error"] = probe_upload_error
            _write_json(status_path, status)
            print(f"Final artifact upload failed: {probe_upload_error}", file=sys.stderr)
            print(
                json.dumps(
                    {
                        "event": "oasis_worker_finished",
                        "mode": mode,
                        "plan_id": plan_id,
                        "job_id": job_id,
                        "status": status["status"],
                        "elapsed_seconds": status["worker_elapsed_seconds"],
                    }
                ),
                flush=True,
            )
            _hold_for_controller()
            return 3
        print(
            json.dumps(
                {
                    "event": "oasis_worker_finished",
                    "mode": mode,
                    "plan_id": plan_id,
                    "job_id": job_id,
                    "status": "complete",
                    "elapsed_seconds": status["worker_elapsed_seconds"],
                }
            ),
            flush=True,
        )
        _hold_for_controller()
        return 0

    arguments = _runner_arguments(output)
    stop_upload = threading.Event()
    upload_interval = int(_required_env("OASIS_UPLOAD_INTERVAL_SECONDS"))

    def upload_periodically() -> None:
        while not stop_upload.wait(upload_interval):
            try:
                sync.upload()
            except Exception as exc:  # the final upload will retry
                print(f"Periodic artifact upload failed: {exc}", file=sys.stderr)

    uploader = threading.Thread(target=upload_periodically, daemon=True)
    if sync.enabled:
        uploader.start()
    process: subprocess.Popen[bytes] | None = None

    def forward_signal(number: int, _frame: Any) -> None:
        if process is not None and process.poll() is None:
            process.send_signal(number)

    previous_term = signal.signal(signal.SIGTERM, forward_signal)
    previous_int = signal.signal(signal.SIGINT, forward_signal)
    return_code = 1
    upload_error: str | None = None
    try:
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(arguments, stdout=stdout, stderr=stderr)
            return_code = process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        stop_upload.set()
        if uploader.is_alive():
            uploader.join(timeout=5)
        finished_at = datetime.now(UTC)
        summary_path = output.with_suffix(".summary.json")
        evaluation_summary = _evaluation_summary(summary_path)
        if return_code == 0 and (
            evaluation_summary is None or evaluation_summary.get("status") != "complete"
        ):
            return_code = 4
        summary_metrics = (
            {
                key: evaluation_summary.get(key)
                for key in (
                    "selected_records",
                    "planned_cells",
                    "completed_cells",
                    "remaining_cells",
                    "correct",
                    "accuracy",
                    "errors",
                    "mean_input_tokens",
                    "mean_output_tokens",
                    "elapsed_seconds",
                )
            }
            if evaluation_summary is not None
            else None
        )
        status = {
            "status": "complete" if return_code == 0 else "failed",
            **common_status,
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": (finished_at - started_at).total_seconds(),
            "worker_elapsed_seconds": (finished_at - container_started_at).total_seconds(),
            "runner_exit_code": return_code,
        }
        if summary_metrics is not None:
            status["evaluation_summary"] = summary_metrics
        if return_code != 0:
            stderr_tail = _read_log_tail(stderr_path)
            status["runner_stderr_tail"] = stderr_tail
            print(
                json.dumps(
                    {
                        "event": "oasis_runner_failed",
                        "plan_id": plan_id,
                        "job_id": job_id,
                        "runner_exit_code": return_code,
                        "stderr_tail": stderr_tail,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
        _write_json(status_path, status)
        try:
            sync.upload()
        except Exception as exc:
            upload_error = f"{type(exc).__name__}: {exc}"
            status["status"] = "artifact_upload_failed"
            status["artifact_upload_error"] = upload_error
            _write_json(status_path, status)
            print(f"Final artifact upload failed: {upload_error}", file=sys.stderr)
        print(
            json.dumps(
                {
                    "event": "oasis_worker_finished",
                    "mode": mode,
                    "plan_id": plan_id,
                    "job_id": job_id,
                    "status": status["status"],
                    "elapsed_seconds": status["worker_elapsed_seconds"],
                    "evaluation_summary": summary_metrics,
                }
            ),
            flush=True,
        )
    _hold_for_controller()
    return return_code if upload_error is None else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="Create a deterministic JSON launch plan.")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--overwrite", action="store_true")
    plan.add_argument("--rows", type=int)
    plan.add_argument("--seed", type=int)
    plan.add_argument("--shards", type=int)
    plan.add_argument("--time-budgets")
    plan.add_argument("--token-budgets")
    plan.add_argument("--gpu-type", action="append", dest="gpu_types")
    plan.add_argument("--gpu-count", type=int)
    plan.add_argument("--model-type", choices=["fake", "transformers"])
    plan.add_argument(
        "--probe-only",
        action="store_true",
        default=None,
        help="Validate image, GPU discovery, and artifact upload without loading a model.",
    )

    launch = commands.add_parser("launch", help="Preview or create Pods from a plan.")
    launch.add_argument("--plan", type=Path, required=True)
    launch.add_argument("--state", type=Path)
    launch.add_argument("--job-id", action="append", default=[])
    launch.add_argument("--max-jobs", type=int)
    launch.add_argument("--print-payloads", action="store_true")
    launch.add_argument("--execute", action="store_true")
    launch.add_argument("--api-base", default=RUNPOD_API_BASE, help=argparse.SUPPRESS)

    status = commands.add_parser("status", help="Query Runpod for Pods in a launch state.")
    status.add_argument("--state", type=Path, required=True)
    status.add_argument("--api-base", default=RUNPOD_API_BASE, help=argparse.SUPPRESS)

    stop = commands.add_parser(
        "stop", help="Stop Pod compute while preserving each Pod and its disk."
    )
    stop.add_argument("--state", type=Path, required=True)
    stop.add_argument("--execute", action="store_true")
    stop.add_argument("--api-base", default=RUNPOD_API_BASE, help=argparse.SUPPRESS)

    cost = commands.add_parser(
        "cost", help="Combine launch state and downloaded job statuses into a cost report."
    )
    cost.add_argument("--state", type=Path, required=True)
    cost.add_argument("--results-root", type=Path, required=True)

    terminate = commands.add_parser(
        "terminate", help="Preview or permanently delete Pods in a launch state."
    )
    terminate.add_argument("--state", type=Path, required=True)
    terminate.add_argument("--execute", action="store_true")
    terminate.add_argument("--api-base", default=RUNPOD_API_BASE, help=argparse.SUPPRESS)

    commands.add_parser("worker", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "worker":
        try:
            return worker_main()
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "oasis_worker_crashed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback_tail": traceback.format_exc()[-16_384:],
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            _hold_for_controller()
            return 1
    try:
        if arguments.command == "plan":
            if arguments.output.exists() and not arguments.overwrite:
                raise ValueError(
                    f"Plan already exists: {arguments.output}; pass --overwrite to replace it"
                )
            plan = build_plan(
                arguments.config,
                PlanOverrides(
                    rows=arguments.rows,
                    seed=arguments.seed,
                    shards=arguments.shards,
                    time_budgets=arguments.time_budgets,
                    token_budgets=arguments.token_budgets,
                    gpu_types=tuple(arguments.gpu_types) if arguments.gpu_types else None,
                    gpu_count=arguments.gpu_count,
                    model_type=arguments.model_type,
                    probe_only=arguments.probe_only,
                ),
            )
            _write_json(arguments.output, plan)
            print(
                f"Wrote {len(plan['jobs'])} jobs for {plan['experiment']['rows_per_dataset']} "
                f"rows/dataset to {arguments.output} (plan {plan['plan_id']})"
            )
            return 0
        if arguments.command == "launch":
            return launch_plan(
                plan_path=arguments.plan,
                state_path=arguments.state,
                execute=arguments.execute,
                requested_ids=arguments.job_id,
                maximum=arguments.max_jobs,
                print_payloads=arguments.print_payloads,
                api_base=arguments.api_base,
            )
        if arguments.command == "status":
            return show_status(state_path=arguments.state, api_base=arguments.api_base)
        if arguments.command == "stop":
            return stop_pods(
                state_path=arguments.state,
                execute=arguments.execute,
                api_base=arguments.api_base,
            )
        if arguments.command == "cost":
            return cost_report(state_path=arguments.state, results_root=arguments.results_root)
        if arguments.command == "terminate":
            return terminate_pods(
                state_path=arguments.state,
                execute=arguments.execute,
                api_base=arguments.api_base,
            )
        parser.error(f"unsupported command: {arguments.command}")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
