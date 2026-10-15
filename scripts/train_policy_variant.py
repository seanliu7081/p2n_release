#!/usr/bin/env python3
"""Train one policy variant on LIBERO or a named real-robot task.

--dry-run resolves configuration only, without reading datasets/checkpoints or
initializing GPUs. Additional Hydra overrides follow -- and take precedence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.real_robot_tasks import REAL_ROBOT_CONFIGS, REAL_ROBOT_TASKS, task_overrides

SIM_CONFIGS = {
    'p2n_ar_classic': 'train_p2n_ar_classic_scratch',
    'p2n_ar_classic_state_history_gate': 'experimental/train_p2n_ar_classic_state_history_gate',
    'p2n_ar_modern': 'experimental/train_p2n_ar_modern_libero',
    'p2n_ar_modern_state_gate': 'experimental/train_p2n_ar_modern_state_gate_libero',
    'p2n_latent_flow': 'train_p2n_latent_flow_resnet18',
    'p2n_state_gate_latent_flow': 'train_p2n_state_gate_latent_flow_resnet18',
    'p2n_action_flow': 'train_p2n_action_flow_resnet18',
    'p2n_state_gate_action_flow': 'train_p2n_state_gate_action_flow_resnet18',
    'diffpolicy': 'train_diffpolicy_libero',
}
TASKS = ('libero', *REAL_ROBOT_TASKS)

def config_name(variant, task):
    return (SIM_CONFIGS if task == 'libero' else REAL_ROBOT_CONFIGS)[variant]

LAUNCHERS = {
    'p2n_ar_modern': 'scripts.train_p2n_ar_modern',
    'p2n_ar_modern_state_gate': 'scripts.train_p2n_ar_modern',
    'p2n_latent_flow': 'scripts.train_p2n_latent_flow',
    'p2n_state_gate_latent_flow': 'scripts.train_p2n_latent_flow',
    'p2n_action_flow': 'scripts.train_p2n_action_flow_resnet18',
    'p2n_state_gate_action_flow': 'scripts.train_p2n_action_flow_resnet18',
}


def compose_config(variant, task, overrides=()):
    if variant not in REAL_ROBOT_CONFIGS:
        raise ValueError(f'Unknown policy variant: {variant}')
    if task not in TASKS:
        raise ValueError(f'Unknown task: {task}')
    if variant in LAUNCHERS:
        return importlib.import_module(LAUNCHERS[variant]).compose_config(variant, task, overrides)
    from hydra import compose, initialize_config_dir
    from src.common.hydra_util import register_new_resolvers
    register_new_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / 'src/config'), version_base=None):
        return compose(config_name=config_name(variant, task),
                       overrides=[*task_overrides(task), *overrides])


def tokenizer_key(variant):
    if variant in ('diffpolicy', 'p2n_action_flow', 'p2n_state_gate_action_flow'):
        return None
    if variant in ('p2n_ar_classic', 'p2n_ar_classic_state_history_gate'):
        return 'policy.action_tokenizer.checkpoint'
    return 'policy.tokenizer_checkpoint'


def main(argv=None):
    from omegaconf import OmegaConf
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index('--') if '--' in argv else len(argv)
    flags, overrides = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--variant', choices=tuple(REAL_ROBOT_CONFIGS))
    selection.add_argument('--config-name', help='Policy config name, with or without experimental/ and .yaml')
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--tokenizer', help='Task-matched OAT checkpoint for AR/latent-flow policies')
    parser.add_argument('--dataset', help='Override the selected task dataset path')
    parser.add_argument('--gpus', help='Physical GPU indices, e.g. 0,1; required for training')
    parser.add_argument('--num-processes', type=int, help='Defaults to the selected GPU count')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path, help='Full artifact for modern AR/flow policies')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--val-batch-size', type=int)
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--grad-accum', type=int)
    parser.add_argument('--num-demo', type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true', help='Check local training inputs without starting training')
    mode.add_argument('--dry-run', action='store_true', help='Print resolved config only; no data/weights/GPU access')
    args = parser.parse_args(flags)
    for field in ('dataset', 'tokenizer', 'output', 'resume'):
        value = getattr(args, field)
        if value is not None:
            resolved = Path(value).expanduser().resolve()
            setattr(args, field, str(resolved) if field in ('dataset', 'tokenizer') else resolved)
    os.chdir(ROOT)
    variant = args.variant
    if args.config_name:
        name = args.config_name.removesuffix('.yaml')
        recipes = SIM_CONFIGS if args.task == 'libero' else REAL_ROBOT_CONFIGS
        matches = [v for v, config in recipes.items() if name in (config, config.split('/')[-1])]
        if not matches:
            parser.error('--config-name must match the selected task; choose one of: ' + ', '.join(recipes.values()))
        variant = matches[0]
    tok_key = tokenizer_key(variant)
    if args.tokenizer and tok_key is None:
        parser.error(f'{variant} does not use an action tokenizer')
    if args.resume and variant not in LAUNCHERS:
        parser.error('--resume is supported here only for modern AR/flow policies')
    if args.resume and args.dry_run:
        parser.error('Use the family launcher for resume preflight; --dry-run here resolves fresh configs only')
    generated = []
    for value, key in ((args.tokenizer, tok_key), (args.dataset, 'task.policy.dataset.zarr_path'),
                       (args.batch_size, 'dataloader.batch_size'), (args.val_batch_size, 'val_dataloader.batch_size'),
                       (args.num_epochs, 'training.num_epochs'),
                       (args.grad_accum, 'training.gradient_accumulate_every'),
                       (args.num_demo, 'training.num_demo')):
        if value is not None:
            generated.append(f'{key}={json.dumps(value)}')
    overrides = [*generated, *overrides]
    # Resume stays with the family launcher, which restores the saved config.
    cfg = None if args.resume else compose_config(variant, args.task, overrides)
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return cfg
    if not args.gpus:
        parser.error('--gpus is required for training')
    gpus = args.gpus.split(',')
    if any(not gpu.isdigit() for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error('--gpus must contain distinct comma-separated integer GPU indices')
    processes = args.num_processes if args.num_processes is not None else len(gpus)
    if processes != len(gpus):
        parser.error('--num-processes must match the number of selected GPUs')
    if cfg is not None and tok_key:
        checkpoint = OmegaConf.select(cfg, tok_key, default=None, throw_on_missing=False)
        if not checkpoint:
            parser.error(f'Task {args.task} needs a matched --tokenizer PATH for {variant}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    if not args.resume and args.output is None:
        args.output = ROOT / 'output/training' / f'{variant}_{args.task}_{stamp}'
    if args.task == 'libero':
        os.environ.setdefault('MUJOCO_GL', 'egl')
        if os.environ['MUJOCO_GL'] in ('egl', 'osmesa'):
            os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])
    if variant in LAUNCHERS:
        family_args = ['--variant', variant, '--task', args.task, '--num-processes', str(processes)]
        family_args += ['--devices' if variant in ('p2n_ar_modern', 'p2n_ar_modern_state_gate') else '--gpus', args.gpus]
        if args.preflight:
            family_args += ['--preflight']
        if args.output:
            family_args += ['--output', str(args.output)]
        if args.resume:
            family_args += ['--resume', str(args.resume)]
        return importlib.import_module(LAUNCHERS[variant]).main([*family_args, '--', *overrides])
    # Classic AR and diffusion retain their established Hydra workspace entrypoint.
    from scripts.p2n_launch_utils import inspect_dataset
    from scripts.action_flow_launch_utils import check_gpu_selection
    check_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if 'offline_validation_enabled' not in check_cfg.training:
        OmegaConf.update(check_cfg, 'training.offline_validation_enabled', True, force_add=True)
    report = inspect_dataset(check_cfg)
    if tok_key and not Path(OmegaConf.select(cfg, tok_key)).expanduser().is_file():
        parser.error('Tokenizer checkpoint does not exist; pass --tokenizer PATH')
    if args.preflight:
        print(json.dumps(report, indent=2))
        return report
    check_gpu_selection(gpus, False)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = (args.output or ROOT / 'output/training' / f'{variant}_{args.task}_{stamp}').expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f'Fresh output directory must be empty: {output}')
    command = [sys.executable]
    if processes > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(processes)]
    command += [str(ROOT / 'scripts/run_workspace.py'), '--config-name=' + config_name(variant, args.task),
                *task_overrides(args.task), *overrides, 'hydra.run.dir=' + json.dumps(str(output))]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


if __name__ == '__main__':
    main()
