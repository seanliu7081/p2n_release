# Past2Next release

This workspace contains ResNet-based robot policies, their training code, and
configuration files. Each policy variant has one shell launcher for both LIBERO
simulation data and recorded real-robot data. Select the domain and dataset with
`--task`; each command starts only the variant named by its launcher.

## Workspace layout

```text
p2n_release/
├── train_<variant>.sh          # Nine policy launchers
├── scripts/
│   ├── train_policy_variant.py # Shared flags and task selection
│   ├── train_p2n_*.py          # Family-specific validation and training
│   └── run_workspace.py       # Hydra entry point, including tokenizer training
├── src/
│   ├── config/                # Policy recipes, dataset schemas, task presets
│   ├── policy/                # Policy implementations
│   ├── model/                 # AR/flow backbones, history encoders, tokenizer helpers
│   ├── perception/            # ResNet encoders
│   ├── dataset/               # Zarr readers and history sampling
│   ├── env_runner/            # Simulator rollout and execution-history handling
│   └── workspace/             # Training loops and checkpoint handling
├── third_party/LIBERO/        # Simulator dependency
├── pyproject.toml / uv.lock   # Package dependencies and locked environment
└── README.md
```

Datasets, tokenizer checkpoints, training outputs, and tests are not bundled.
Training creates `output/training/` as needed. Old experiment shell scripts,
implementation plans, and verification reports have been removed. Upstream flow
reference attribution is retained in `src/model/flow/upstream_reference.json`.

## Code and environment setup

Run examples from `/workspace/p2n_release` (or your local checkout path).
The supplied lockfile supports the Python 3.10 environment used for this release.
The observation encoder requires `robomimic==0.2.0`; the data code uses Zarr 2.

On this instance, use the existing environment without reinstalling packages:

```bash
cd /workspace/p2n_release
export TRAIN_PY=/venv/oat/bin/python
export PYTHONPATH="$PWD:$PWD/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
"$TRAIN_PY" -c 'import torch, src, libero; print(torch.__version__); print(src.__file__); print(libero.__file__)'
```

For a separate environment from the checked-out source, with `uv` installed:

```bash
git clone --recurse-submodules https://github.com/seanliu7081/p2n_release.git
cd p2n_release
uv sync --frozen --python 3.10
source .venv/bin/activate
export TRAIN_PY="$PWD/.venv/bin/python"
python -c 'import torch, torchvision, zarr, robomimic, src, libero; print(torch.__version__, zarr.__version__); print(torch.cuda.is_available())'
```

Use a PyTorch CUDA build compatible with your GPU and host driver. LIBERO rendering
also needs working EGL/OpenGL libraries, or OSMesa for software rendering. For
headless simulator rollouts:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Launchers use `TRAIN_PY` when set, otherwise `python` from the active environment.
They resolve their source directory automatically. File paths supplied through
`--dataset`, `--tokenizer`, `--output`, and `--resume` are resolved relative to the
calling directory. Relative paths inside YAML/Hydra overrides use the release root.
`environments/real_robot/*.txt` are historical environment inventories, not portable
installation requirements. Use the lockfile for a fresh environment.

## Policy launchers

All nine launchers support `--task libero` and all real-robot tasks below.
All visual encoders are trainable ResNet-18 encoders initialized from scratch.

| Launcher | Policy | Frozen OAT tokenizer required? |
|---|---|---|
| `train_p2n_ar_classic.sh` | Classic autoregressive Past2Next | Yes |
| `train_p2n_ar_classic_state_history_gate.sh` | Classic AR with gated state/action history | Yes |
| `train_p2n_ar_modern.sh` | Modern autoregressive transformer | Yes |
| `train_p2n_ar_modern_state_gate.sh` | Modern AR with gated state/action history | Yes |
| `train_p2n_latent_flow.sh` | Flow over OAT latent codes | Yes |
| `train_p2n_state_gate_latent_flow.sh` | Latent flow with gated state/action history | Yes |
| `train_p2n_action_flow.sh` | Flow directly over continuous actions | No |
| `train_p2n_state_gate_action_flow.sh` | Direct action flow with gated state/action history | No |
| `train_diffpolicy.sh` | Diffusion action baseline | No |

There are nine launchable variants and ten configured policy classes: the classic
state-gated variant uses a separate rotation-6D adapter for real-robot observations.
No standalone ungated state-history policy remains. Base policies still receive
current robot state and, for Past2Next, past actions.

Modern AR uses a 16-layer, 768-wide, 12-head transformer with SwiGLU. Latent flow
uses DiT-X with the same depth/width/heads and SwiGLU, generating eight 5D OAT
latent slots. Direct action flow uses DiT-X with GELU, generating sixteen 7D
actions. The classic scratch recipe uses eight layers, width 256, and eight heads;
diffusion uses four layers, width 256, and four heads. See `src/config/` for the
complete training and architecture settings.

Policy training configs are grouped by variant under
`src/config/policy/<variant>/`. Each of the nine folders contains `sim.yaml`
and `real.yaml`. Shared recipe defaults use `base.yaml` where needed; additional
classic AR recipes stay in the `p2n_ar_classic/` folder. Task schemas remain in
`src/config/task/`, and real task presets remain in `src/config/real_robot_task/`.

For example, direct Hydra selection uses
`--config-name=policy/p2n_action_flow/sim` or
`--config-name=policy/p2n_action_flow/real`. The shell commands and `--task`
selection are unchanged.

## Tasks and datasets

`--task libero` selects LIBERO-10, with the configured 500-demonstration dataset.
The real-robot presets are stored once in `src/config/real_robot_task/`:

| Task | Default dataset under `/workspace/ysk/zarr/` | Demos | Validation fraction |
|---|---|---:|---:|
| `nut_washer` | `nut_washer_v3_N77.zarr` | 77 | 5% |
| `nut_washer_n62` | `nut_washer_N62.zarr` | 62 | 10% |
| `pen_cabinet` | `pen_cabinet_N67.zarr` | 67 | 10% |
| `pen_cabinet_lp3` | `pen_cabinet_lp3_N67.zarr` | 67 | 10% |
| `fruits` | `fruits_N51.zarr` | 51 | 10% |
| `fruits_v2` | `fruits_v2_N49.zarr` | 49 | 10% |

Use `--dataset /path/to/data.zarr` to override a location. If the episode count
changes, also set `--num-demo N`; changing the split requires a matching tokenizer
for policies that use OAT. Real-robot presets use split seed 42.

The common schema uses two 128×128 RGB cameras (`agentview_rgb` and
`robot0_eye_in_hand_rgb`), end-effector position, orientation, gripper state,
`task_uid`, and 7D actions. Simulation uses quaternion orientation; real-robot data
uses rotation-6D rows and measured gripper width. Zarr arrays live under `data/`
with episode boundaries in `meta/episode_ends`. Dataset adapters derive history
windows and validity masks from these recordings.

Real-robot actions follow the existing dataset convention at 30 Hz: base-frame
position deltas in metres, end-effector-frame rotation-vector deltas in radians
(`R_next = R_previous @ Exp(drotvec)`), and an absolute gripper command from 0
(open) to 1 (closed). Observed gripper width is in millimetres. Training and
held-out validation read recorded data; these launchers do not control a physical robot.

## Training

Inspect a configuration first. `--dry-run` performs no data/checkpoint reads,
model construction, GPU initialization, logging, or output-directory creation:

```bash
bash train_p2n_ar_modern_state_gate.sh --task libero --dry-run
bash train_p2n_state_gate_action_flow.sh --task fruits_v2 --dry-run
```

Train the same variant on simulation or recorded real-robot data:

```bash
# Simulation: frozen tokenizer must match this LIBERO dataset and split.
bash train_p2n_ar_modern_state_gate.sh --task libero --gpus 0,1 \
  --dataset /path/to/libero10_N500.zarr --tokenizer /path/to/libero_oat.ckpt

# Real robot: direct action flow needs no tokenizer.
bash train_p2n_state_gate_action_flow.sh --task fruits_v2 --gpus 0,1 \
  --dataset /path/to/fruits_v2_N49.zarr --batch-size 4 --grad-accum 8

# Real robot: classic AR uses the same task selection.
bash train_p2n_ar_classic.sh --task fruits_v2 --gpus 0 \
  --dataset /path/to/fruits_v2_N49.zarr --tokenizer /path/to/fruits_v2_oat.ckpt
```

Common flags:

| Flag | Meaning |
|---|---|
| `--task NAME` | Required simulation/robot task preset |
| `--gpus 0,1` | Explicit GPU indices; required for training and preflight |
| `--tokenizer PATH` | Frozen, task-matched OAT checkpoint |
| `--dataset PATH`, `--num-demo N` | Dataset location and expected episode count |
| `--batch-size N`, `--val-batch-size N` | Per-process loader batches |
| `--grad-accum N` | Microbatches per optimizer update |
| `--num-epochs N` | Training epochs |
| `--output PATH` | Fresh run directory; default is timestamped under `output/training/` |
| `--dry-run` | Resolve and print config only |
| `--preflight` | Run family-specific local input checks without training |
| `--resume PATH` | Full checkpoint resume for modern AR and flow families |
| `--help` | Show the shared options |

Preflight depth depends on the family: classic AR/diffusion check dataset metadata
and tokenizer-file existence where needed; modern AR and flow use their existing
source/schema checks. A successful preflight does not establish GPU training or
robot rollout performance. Classic AR/diffusion checkpoint handling remains
available through `scripts/run_workspace.py`; the common `--resume` flag is only
supported for modern AR/flow.

Flow training batches must be at least four and divisible by four. Effective
batch size is `batch-size × number-of-GPUs × grad-accum`. No launcher implicitly
starts both the base and gated variant. Hydra overrides after `--` take precedence
above convenience flags:

```bash
bash train_p2n_action_flow.sh --task libero --gpus 0,1 \
  --dataset /path/to/libero10_N500.zarr --batch-size 4 --grad-accum 8 \
  -- task.policy.lazy_eval=false logging.mode=offline training.rollout_every=25
```

Modern AR simulation recipes enable simulator rollouts by default. Classic AR,
flow, and diffusion recipes default to lazy evaluation; set
`task.policy.lazy_eval=false` to enable their simulator rollouts. Real-robot tasks
use offline validation. Logging mode,
checkpoint cadence, and other training defaults come from the selected recipe.
For fresh AR/latent-flow training, pass `--tokenizer` explicitly when preset paths
are unavailable. `fruits`, `fruits_v2`, and `nut_washer_n62` have no preset OAT path.

## Training an OAT tokenizer

Use the Hydra workspace entry point for the tokenizer stage:

```bash
"${TRAIN_PY:-python}" scripts/run_workspace.py --config-name=train_oattok_so3aug \
  task/tokenizer=real_robot/fruits_v2 \
  task.tokenizer.dataset.zarr_path=/path/to/fruits_v2_N49.zarr \
  training.num_demo=49 task.tokenizer.dataset.val_ratio=0.1 \
  hydra.run.dir=output/training/fruits_v2_tokenizer
```

For simulation, use `task/tokenizer=libero/libero10` and the corresponding dataset.
Match the policy's action horizon (16), dataset, episode split, and split seed when
training its tokenizer. Pass the resulting checkpoint to the policy launcher.
The six-task presets apply to policy launchers; direct tokenizer training requires
explicit dataset/split overrides when using a different dataset version.

Renamed AR modules use `p2n_ar_classic`, `p2n_ar_modern`, and
`p2n_ar_modern_state_gate`. Checkpoints recording former module paths or variant IDs
need configuration/metadata migration; cleanup does not rewrite saved checkpoints.

The Python package lives in `src/`; imports and Hydra targets use `src.*`.
The distribution and existing environment are still named `oat`. Reinstall
an editable installation after updating an existing checkout. The release
checkpoint loaders translate legacy `oat.*` Hydra targets to `src.*` in
memory, including embedded tokenizer configurations; checkpoint files and
external data paths are unchanged. This supports the package rename only,
not previously removed architectures or renamed variant IDs. Full pickled
workspace snapshots require their original code environment.
