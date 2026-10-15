"""Shared real-robot task selection for all release policy launchers."""
from pathlib import Path
from omegaconf import OmegaConf

# Dataset action convention shared by the real-robot tasks.
ACTION_SEMANTICS = {
    "fps": 30,
    "translation": {"slice": [0, 3], "representation": "delta_position",
                    "frame": "base", "units": "metres",
                    "integration": "p_next = p_previous + dp"},
    "rotation": {"slice": [3, 6], "representation": "rotation_vector",
                 "frame": "end_effector", "units": "radians",
                 "integration": "R_next = R_previous @ Exp(drotvec)"},
    "gripper": {"index": 6, "representation": "absolute_command",
                "open": 0.0, "closed": 1.0},
    "gripper_observation": {"key": "robot0_gripper_qpos",
                            "representation": "measured_width",
                            "units": "millimetres", "range": [0, 80]},
}

ROOT = Path(__file__).resolve().parents[1]
TASK_PRESETS = {
    path.stem: OmegaConf.load(path)
    for path in sorted((ROOT / 'src/config/real_robot_task').glob('*.yaml'))
}
REAL_ROBOT_TASKS = tuple(TASK_PRESETS)
TASK_TYPES = {'libero': 'libero', 'real_robot': 'real_robot',
              **{name: 'real_robot' for name in REAL_ROBOT_TASKS}}
TASK_NAMES = {name: preset.task_name for name, preset in TASK_PRESETS.items()}
TASK_NAMES['real_robot'] = TASK_NAMES['nut_washer']

REAL_ROBOT_CONFIGS = {
    'p2n_ar_classic': 'experimental/train_p2n_ar_classic_real_robot',
    'p2n_ar_modern': 'experimental/train_p2n_ar_modern_real_robot',
    'p2n_ar_modern_state_gate': 'experimental/train_p2n_ar_modern_state_gate_real_robot',
    'p2n_latent_flow': 'experimental/train_p2n_latent_flow_resnet18_real_robot',
    'p2n_state_gate_latent_flow': 'experimental/train_p2n_state_gate_latent_flow_resnet18_real_robot',
    'p2n_action_flow': 'experimental/train_p2n_action_flow_resnet18_real_robot',
    'p2n_state_gate_action_flow': 'experimental/train_p2n_state_gate_action_flow_resnet18_real_robot',
    'p2n_ar_classic_state_history_gate': 'experimental/train_p2n_ar_classic_state_history_gate_real_robot',
    'diffpolicy': 'train_diffpolicy_real_robot',
}


def task_overrides(task):
    if task not in TASK_TYPES:
        raise ValueError(f'Unknown task: {task}')
    if task == 'libero':
        return []
    return [f"real_robot_task={'nut_washer' if task == 'real_robot' else task}"]


def validate_task_selection(cfg, task):
    """Prevent a task flag or resume from silently selecting another dataset recipe."""
    expected_type = TASK_TYPES.get(task)
    if expected_type is None or cfg.task_type != expected_type or cfg.policy.task != expected_type:
        raise ValueError('Selected task and policy task must agree')
    expected_name = TASK_NAMES.get(task)
    if expected_name is not None and cfg.task.policy.task_name != expected_name:
        raise ValueError(f'Selected task {task!r} requires recipe {expected_name!r}, '
                         f'received {cfg.task.policy.task_name!r}')
    return expected_type


def selected_task(cfg):
    """Recover the named task when a worker loads an already resolved config."""
    if cfg.task_type == 'libero':
        return 'libero'
    for name, expected in TASK_NAMES.items():
        if cfg.task.policy.task_name == expected:
            return name
    raise ValueError(f'Unknown real-robot task: {cfg.task.policy.task_name}')
