"""Shared configuration, device and source checks for ResNet policy launchers."""
from pathlib import Path
import csv
import io
import json
import math
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]


def inspect_dataset(cfg):
    # Reuse the existing read-only schema/split inspector; it does not construct a policy.
    from scripts.p2n_launch_utils import inspect_dataset as inspect_existing_dataset
    import zarr
    from src.dataset.action_flow_dataset import dataset_fingerprint
    split = inspect_existing_dataset(cfg)
    ds = cfg.task.policy.dataset
    root = zarr.open(str(Path(ds.zarr_path).expanduser()), mode="r")
    arrays = {key: root["data"][key] for key in sorted(set([ds.action_key, *ds.obs_keys]))}
    split["identity"] = dataset_fingerprint(ds.zarr_path, root["meta"]["episode_ends"][:], arrays, ds.action_key)
    split["sample_id_definition"] = "absolute_action_anchor_v1"
    split["future_action_valid_definition"] = "pad_before + arange(horizon) in [sample_start,sample_end)"
    return split

def inspect_training_normalizer(cfg, split):
    """Fit only the small numeric training replay arrays, never sampled windows.

    This mirrors the existing dataset normalizer math without loading cameras or
    instantiating a policy. Its result is a CPU preflight report, not a substitute
    for the dataset-owned normalizer used by fresh training.
    """
    import numpy as np
    import torch
    import zarr
    from src.model.common.normalizer import LinearNormalizer
    ds = cfg.task.policy.dataset
    root = zarr.open(str(Path(ds.zarr_path).expanduser()), mode="r")
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    lengths = np.diff(np.r_[0, ends])
    train_mask = np.zeros(len(ends), dtype=bool)
    train_mask[split["train_episode_ids"]] = True
    frame_mask = np.repeat(train_mask, lengths)
    if not frame_mask.any():
        raise ValueError("Normalizer requires nonempty training replay frames")
    fields = {"action": ds.action_key, **{key: key for key, spec in cfg.shape_meta.obs.items() if spec.type != "rgb"}}
    normalizer = LinearNormalizer()
    for name, key in fields.items():
        data = np.asarray(root["data"][key][:])[frame_mask]
        if not np.isfinite(data).all():
            raise ValueError(f"Training replay field {key} contains non-finite values")
        normalizer.fit({name: data}, last_n_dims=1, mode=cfg.normalization.mode)
        field = normalizer[name]
        scale, offset = field.params_dict["scale"], field.params_dict["offset"]
        if not torch.isfinite(scale).all() or not torch.isfinite(offset).all() or (scale == 0).any():
            raise ValueError(f"Training normalizer for {name} must be finite and invertible")
        original = torch.as_tensor(data, dtype=torch.float32)
        recovered = field.unnormalize(field.normalize(original))
        if not torch.isfinite(recovered).all() or not torch.allclose(recovered, original, atol=2e-5, rtol=2e-5):
            raise ValueError(f"Training normalizer round trip failed for {name}")
    action = normalizer["action"]
    return {"source": "training_replay_frames", "mode": cfg.normalization.mode,
            "data_path": str(Path(ds.zarr_path).expanduser().resolve()),
            "dataset_identity": split["identity"], "action_key": ds.action_key,
            "train_episode_ids": split["train_episode_ids"], "train_frames": int(frame_mask.sum()),
            "fields": list(fields), "overlapping_windows": False, "padded_frames": False,
            "refit_on_resume": False,
            "action_scale": action.params_dict["scale"].tolist(),
            "action_offset": action.params_dict["offset"].tolist(),
            "round_trip": "passed"}

def check_gpu_selection(gpus, allow_busy=False):
    """Read NVIDIA inventory immediately before launch, without a CUDA context."""
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True)
    inventory = {}
    for row in result.stdout.splitlines():
        index, memory, utilization = [part.strip() for part in row.split(",")]
        inventory[index] = (int(memory), int(utilization))
    missing = set(gpus) - set(inventory)
    if missing:
        raise ValueError(f"Selected GPU indices do not exist: {sorted(missing)}")
    busy = {gpu: inventory[gpu] for gpu in gpus if inventory[gpu][0] > 1024 or inventory[gpu][1] > 5}
    if busy and not allow_busy:
        raise RuntimeError(f"Selected GPUs appear busy (memory MiB, utilization %): {busy}; choose free GPUs or explicitly use --allow-busy-gpus")
