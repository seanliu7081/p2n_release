"""Shared configuration, device and source checks for ResNet policy launchers."""
from src.common.checkpoint_compat import load_checkpoint
from pathlib import Path
import csv
import io
import json
import math
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]


def validate_original_obs_config(cfg, variant, task):
    gate = variant == "p2n_ar_modern_state_gate"
    target = "src.policy.p2n_ar_modern." + ("P2NARModernStateGatePolicy" if gate else "P2NARModernPolicy")
    if cfg.variant != variant or cfg.policy.variant != variant or cfg.policy._target_ != target:
        raise ValueError("Selected variant, policy target and explicit variant must agree")
    if cfg.task_type != task or cfg.policy.task != task:
        raise ValueError("Selected task and policy task must agree")
    if cfg.training.get("init_checkpoint"):
        raise ValueError("Old policy initialization is incompatible; use fresh initialization or explicit resume")
    if cfg.training.resume and not cfg.training.get("resume_checkpoint"):
        raise ValueError("Resume requires --resume CHECKPOINT or training.resume_checkpoint")
    if cfg.task.policy.dataset.history_padding != "zero" or cfg.task.policy.dataset.return_history_validity is not True:
        raise ValueError("Both variants require zero history padding and explicit validity")
    dataset_target = cfg.task.policy.dataset._target_
    expected_dataset = {
        (False, "libero"): "src.dataset.zarr_dataset_with_prev_window.ZarrDatasetWithPrevWindow",
        (True, "libero"): "src.dataset.zarr_dataset_with_state_history.ZarrDatasetWithStateHistory",
        (False, "real_robot"): "src.dataset.real_robot_dataset.RealRobotZarrDatasetWithPrevWindow",
        (True, "real_robot"): "src.dataset.real_robot_state_history.RealRobotZarrDatasetWithStateHistory",
    }[gate, task]
    if dataset_target != expected_dataset:
        raise ValueError(f"Variant/task requires dataset {expected_dataset}")
    if not gate and any(key.startswith("history_") or key.startswith("state_history") for key in cfg.policy):
        raise ValueError("p2n_ar_modern must not construct state history or gate modules")
    if gate and (cfg.policy.state_history_steps != cfg.past_n + 1 or cfg.policy.history_summary_tokens != 4):
        raise ValueError("Gate recipe requires past_n + 1 measured states and four summaries")
    if task == "real_robot":
        if cfg.task.policy.env_runner is not None:
            raise ValueError("Real robot uses offline validation and no simulator runner")
    else:
        expected_runner = "P2NARModernStateGateLiberoRunner" if gate else "P2NARModernLiberoRunner"
        if not cfg.task.policy.env_runner._target_.endswith("." + expected_runner):
            raise ValueError("LIBERO runner does not match execution/state-history capabilities")
    for key in ("num_epochs", "gradient_accumulate_every", "checkpoint_every", "val_every", "sample_every"):
        if int(cfg.training[key]) < 1:
            raise ValueError(f"training.{key} must be positive")

def inspect_dataset(cfg):
    """Read Zarr schema and episode metadata without copying image arrays into RAM."""
    import numpy as np
    import zarr
    from src.common.seq_sampler import get_val_mask, downsample_mask
    ds = cfg.task.policy.dataset
    path = Path(ds.zarr_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset is missing: {path}")
    root = zarr.open(str(path), mode="r")
    arrays = root["data"]
    ends = np.asarray(root["meta"]["episode_ends"][:])
    if ends.ndim != 1 or not len(ends) or (np.diff(np.r_[0, ends]) <= 0).any():
        raise ValueError("Dataset episode boundaries must be increasing")
    action = arrays[ds.action_key]
    if tuple(action.shape[1:]) != tuple(cfg.shape_meta.action.shape) or action.shape[0] != ends[-1]:
        raise ValueError("Dataset action shape/episode count does not match policy schema")
    for key, spec in cfg.shape_meta.obs.items():
        if key not in ds.obs_keys or key not in arrays:
            raise ValueError(f"Dataset is missing required observation {key}")
        value = arrays[key]
        if tuple(value.shape[1:]) != tuple(spec.shape) or value.shape[0] != ends[-1]:
            raise ValueError(f"Dataset observation schema mismatch: {key}")
        if spec.type == "rgb" and value.dtype != np.dtype("uint8"):
            raise ValueError(f"RGB port {key} must be uint8 byte-range RGB")
    if int(cfg.training.num_demo) != len(ends):
        raise ValueError(f"training.num_demo={cfg.training.num_demo} differs from {len(ends)} actual episodes")
    held_out = get_val_mask(len(ends), float(ds.val_ratio), int(ds.seed))
    train = downsample_mask(~held_out, ds.get("max_train_episodes"), int(ds.seed))
    # Existing validation views use the complement of the actual training mask.
    validation = ~train
    if cfg.training.offline_validation_enabled and not validation.any():
        raise ValueError("Offline validation is enabled but the actual held-out split is empty")
    lengths = np.diff(np.r_[0, ends])
    return {"schema_version": 1, "data_path": str(path), "val_ratio": float(ds.val_ratio),
            "seed": int(ds.seed), "train_episode_ids": np.flatnonzero(train).tolist(),
            "validation_episode_ids": np.flatnonzero(validation).tolist(),
            "train": {"episodes": int(train.sum()), "windows": int(lengths[train].sum())},
            "validation": {"episodes": int(validation.sum()), "windows": int(lengths[validation].sum())},
            "offline_validation_enabled": bool(cfg.training.offline_validation_enabled)}

def validate_tokenizer_source(cfg):
    import dill
    import torch
    path = cfg.policy.tokenizer_checkpoint
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"Fresh training requires a task-matched tokenizer checkpoint: {path}")
    payload = load_checkpoint(path, map_location="cpu", pickle_module=dill, weights_only=False)
    saved = payload.get("cfg")
    if saved is None or "tokenizer" not in saved.get("task", {}):
        raise ValueError("Tokenizer checkpoint lacks task/normalizer provenance metadata")
    source = saved.task.tokenizer.dataset
    target = cfg.task.policy.dataset
    for key in ("zarr_path", "action_key", "n_action_steps", "seed", "val_ratio", "max_train_episodes"):
        if source.get(key) != target.get(key):
            raise ValueError(f"Tokenizer action/normalizer source mismatch: dataset.{key}")
    if "ema_model" not in payload.get("state_dicts", {}):
        raise ValueError("Fresh training requires the specified tokenizer EMA weights")

def check_gpu_idle(devices=None, required_devices=1, *, allowed_pids=()):
    """Read nvidia-smi only; never terminate or otherwise change existing jobs.

    Numeric selections are nvidia-smi indices. Callers use the returned UUIDs
    to remove ambiguity with CUDA's default device ordering.
    """
    def query(fields, kind):
        result = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        return [[item.strip() for item in row] for row in csv.reader(io.StringIO(result.stdout)) if row]

    try:
        rows = query("index,uuid,name", "gpu")
        processes = query("gpu_uuid,pid,process_name", "compute-apps")
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Cannot verify idle GPUs with nvidia-smi before execution") from error
    gpu_by_index = {row[0]: {"index": row[0], "uuid": row[1], "name": row[2]} for row in rows}
    mask = devices if devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    requested = list(gpu_by_index) if mask is None else [item.strip() for item in mask.split(",") if item.strip()]
    selected = []
    for value in requested:
        if value in gpu_by_index:
            selected.append(gpu_by_index[value])
        else:
            matches = [gpu for gpu in gpu_by_index.values() if gpu["uuid"].startswith(value)]
            if len(matches) != 1:
                raise ValueError(f"Cannot identify selected GPU {value!r}; use an index or unique GPU UUID")
            selected.append(matches[0])
    if len({gpu["uuid"] for gpu in selected}) != len(selected):
        raise ValueError("Selected GPU mask contains duplicate devices")
    if len(selected) < required_devices:
        raise ValueError(f"Requested {required_devices} processes but only {len(selected)} GPUs are visible")
    selected = selected[:required_devices]
    selected_uuids = {gpu["uuid"] for gpu in selected}
    allowed = {str(pid) for pid in allowed_pids}
    busy = [{"gpu_uuid": uuid, "pid": pid, "process": name}
            for uuid, pid, name in processes if uuid in selected_uuids and pid not in allowed]
    if busy:
        raise RuntimeError(f"Selected GPUs already have compute jobs: {json.dumps(busy)}")
    return selected

def configure_training_devices(selected, *, render=True):
    """Keep precise CUDA UUID selection and map the renderer independently.

    EGL indices can differ from CUDA ordinals. The isolated probe enumerates
    device metadata without opening rendering contexts. Simulator wrappers use
    the selected EGL index only while constructing/forking their environments.
    """
    renderer = None
    if render and os.environ.get("MUJOCO_GL", "egl") == "egl":
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        result = subprocess.run(
            [sys.executable, "-m", "src.common.libero_egl_devices"],
            cwd=str(ROOT), env=env, text=True, capture_output=True, check=True,
        )
        devices = json.loads(result.stdout.strip())
        uuid = selected[0]["uuid"].removeprefix("GPU-").lower()
        matches = [device for device in devices
                   if device["uuid"].removeprefix("GPU-").lower() == uuid]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot identify one EGL renderer for GPU {selected[0]['index']}")
        renderer = matches[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu["uuid"] for gpu in selected)
    # Installed robosuite compares CUDA and EGL indices at import time. The
    # scoped simulator wrapper supplies matching renderer-only values instead.
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    os.environ.pop("P2N_LIBERO_EGL_DEVICE_ID", None)
    if renderer is not None:
        os.environ["P2N_LIBERO_EGL_DEVICE_ID"] = str(renderer["egl_device_id"])
    return {"training_gpus": selected, "renderer": renderer}

def inspect_simulator(cfg):
    """Check imports and assets without constructing an environment or renderer."""
    if cfg.task.policy.lazy_eval:
        return {"enabled": False}
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero")))
    if not (config_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"LIBERO configuration is missing: {config_dir / 'config.yaml'}")
    import hydra
    import importlib.util
    for package in ("libero", "robosuite", "mujoco", "OpenGL", "gymnasium"):
        if importlib.util.find_spec(package) is None:
            raise ImportError(f"LIBERO evaluation dependency is missing: {package}")
    from libero.libero import get_libero_path, benchmark

    hydra.utils.get_class(cfg.task.policy.env_runner._target_)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    bddl_root = Path(get_libero_path("bddl_files"))
    assets = Path(get_libero_path("assets"))
    if not assets.is_dir():
        raise FileNotFoundError(f"LIBERO assets are missing: {assets}")
    for index in range(suite.n_tasks):
        task = suite.get_task(index)
        path = bddl_root / task.problem_folder / task.bddl_file
        if not path.is_file():
            raise FileNotFoundError(f"LIBERO task definition is missing: {path}")
        if cfg.task.policy.env_runner.protocol == "official":
            path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            if not path.is_file():
                raise FileNotFoundError(f"LIBERO initial states are missing: {path}")
    return {
        "enabled": True, "tasks": suite.n_tasks,
        "rollout_every": int(cfg.training.rollout_every),
        "epoch_labels": "0, interval, 2*interval, ... (after each labeled epoch)",
        "episodes_per_evaluation": int(cfg.task.policy.env_runner.n_test),
        "parallel_envs": min(int(cfg.task.policy.env_runner.n_parallel_envs),
                             int(cfg.task.policy.env_runner.n_test)),
        "protocol": cfg.task.policy.env_runner.protocol,
        "assets": str(assets), "status": "Dependencies/assets checked; simulator rollout not run",
    }
