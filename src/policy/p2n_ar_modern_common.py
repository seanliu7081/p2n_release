"""Shared history, training and execution contracts for ResNet Past2Next policies."""
from __future__ import annotations

import contextlib
import copy
import hashlib

import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

from src.policy.base_policy import BasePolicy


def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return copy.deepcopy(value)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bool_mask(value, shape, name, device):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must have shape {tuple(shape)}')
    if value.device != device:
        raise ValueError(f'{name} must be on {device}')
    if value.dtype != torch.bool:
        if not ((value == 0) | (value == 1)).all():
            raise ValueError(f'{name} must contain boolean or exact zero/one flags')
        value = value.bool()
    return value


class P2NARModernCommonPolicy(BasePolicy):
    """One context path for training, self-past, offline and acknowledged rollout."""

    supports_explicit_past_actions = True
    supports_explicit_past_action_valid = True
    supports_generated_history_validation = True
    requires_execution_acknowledgement = True
    requires_state_history = False
    supports_history_summary_gate = False
    VARIANT = 'p2n_ar_modern'
    CONTEXT_SCHEMA_VERSION = 1


    def train(self, mode=True):
        super().train(mode)
        self.action_tokenizer.eval()
        return self

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if not strict:
            raise ValueError('New policy checkpoints require strict loading')
        if '_variant_code' not in state_dict or int(state_dict['_variant_code']) != int(self.requires_state_history):
            raise ValueError('Checkpoint variant does not match this policy')
        if int(state_dict.get('_context_schema', -1)) != self.CONTEXT_SCHEMA_VERSION:
            raise ValueError('Checkpoint ContextBatch schema does not match')
        result = super().load_state_dict(state_dict, strict=True, **kwargs)
        self.action_tokenizer.requires_grad_(False).eval()
        self.reset()
        return result

    def set_normalizer(self, normalizer):
        if isinstance(normalizer, (list, tuple)):
            if len(normalizer) != 1:
                raise ValueError('Expected one task normalizer')
            normalizer = normalizer[0]
        required = ['action', *self.obs_encoder.state_ports]
        missing = [k for k in required if k not in normalizer.params_dict]
        if missing:
            raise KeyError(f'Missing training-set normalizer fields: {missing}')
        self.action_normalizer.load_state_dict(normalizer.state_dict())
        self.action_normalizer.requires_grad_(False)
        # The tokenizer retains its checkpoint's own action normalizer.

    def _safe_past(self, past_actions, past_action_valid):
        if (not isinstance(past_actions, torch.Tensor) or past_actions.ndim != 3
                or tuple(past_actions.shape[1:]) != (self.past_n, self.action_dim)
                or not past_actions.is_floating_point()):
            raise ValueError(f'past_actions must be floating [B,{self.past_n},{self.action_dim}]')
        valid = bool_mask(past_action_valid, past_actions.shape[:2], 'past_action_valid', past_actions.device)
        if not torch.isfinite(past_actions[valid]).all():
            raise ValueError('Valid historical commands must be finite')
        safe = torch.where(valid[..., None], past_actions, torch.zeros_like(past_actions))
        normalized = self.action_normalizer['action'].normalize(safe)
        return torch.where(valid[..., None], normalized, torch.zeros_like(normalized)), valid


    @property
    def self_past_step(self):
        return int(self._self_past_optimizer_step.item())

    def set_self_past_step(self, step):
        if step < 0:
            raise ValueError('Optimizer update count must be nonnegative')
        self._self_past_optimizer_step.fill_(step)

    def on_optimizer_step(self):
        if self.training:
            self._self_past_optimizer_step.add_(1)

    def self_past_probability(self):
        progress = self.self_past_step - self.self_past_warmup_steps
        if progress < 0:
            return 0.0
        return self.self_past_p * (min(1., progress / self.self_past_ramp_steps)
                                   if self.self_past_ramp_steps else 1.)

    @contextlib.contextmanager
    def _rollout_mode(self):
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            yield
        finally:
            # Direct restoration preserves intentionally mixed train/eval modes.
            for module, mode in modes:
                module.training = mode
            self.action_tokenizer.eval()

    @contextlib.contextmanager
    def _clean_autocast_cache(self):
        torch.clear_autocast_cache()
        try:
            yield
        finally:
            torch.clear_autocast_cache()

    def _generate_actions(self, obs, past, valid, n_tokens, temperature, topk):
        context = self.build_context(obs, past, valid)
        tokens = self.model.generate(context, max_new_tokens=n_tokens,
                                     temperature=temperature, top_k=topk)
        return self.action_tokenizer.detokenize(tokens=tokens)

    def _maybe_self_past(self, batch, past, probability=None):
        probability = self.self_past_probability() if probability is None else probability
        if probability <= 0:
            return past
        required = ['prev_obs', 'prev_past_action', 'prev_past_action_valid', 'prev_window_valid']
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f'Self-past needs previous-window metadata: {missing}')
        valid_windows = bool_mask(batch['prev_window_valid'].reshape(-1), (past.shape[0],),
                                  'prev_window_valid', past.device)
        indices = valid_windows.nonzero(as_tuple=True)[0]
        if not len(indices):
            return past
        generated = past.detach().clone()
        # Previous-window generation finishes before the current training graph exists.
        with self._rollout_mode(), self._clean_autocast_cache():
            for index in indices.split(self.self_past_chunk_size):
                with torch.inference_mode():
                    previous = batch['prev_past_action'][index]
                    pred = self._generate_actions(
                        {key: value[index] for key, value in batch['prev_obs'].items()},
                        previous, batch['prev_past_action_valid'][index], self.max_seq_len,
                        self.self_past_temperature, self.self_past_topk,
                    )
                    chunk = torch.cat((previous, pred[:, :self.n_action_steps]), dim=1)[:, -self.past_n:]
                with torch.inference_mode(False):
                    chunk = chunk.detach().clone().to(dtype=past.dtype)
                generated[index] = chunk
        mask = bool_mask(batch['past_action_valid'], past.shape[:2], 'past_action_valid', past.device)
        use_self = valid_windows[:, None] & mask
        if probability < 1:
            use_self = use_self & (torch.rand(past.shape[0], 1, device=past.device) < probability)
        return torch.where(use_self[..., None], generated, past)

    def forward(self, batch, history_mode=None):
        history_mode = ('configured' if self.training else 'expert') if history_mode is None else history_mode
        if history_mode not in ('expert', 'generated', 'configured'):
            raise ValueError('history_mode must be expert, generated, or configured')
        if 'past_action_valid' not in batch:
            raise KeyError('New policies require top-level past_action_valid')
        past = batch['past_action']
        if history_mode != 'expert':
            past = self._maybe_self_past(batch, past, 1. if history_mode == 'generated' else None)
        with torch.no_grad():
            targets = self.action_tokenizer.tokenize(batch['action']).long()
        if targets.shape != (past.shape[0], self.max_seq_len):
            raise ValueError('Tokenizer returned an unexpected latent shape')
        context = self.build_context(batch['obs'], past, batch['past_action_valid'])
        bos = torch.full((past.shape[0], 1), self.bos_id, dtype=torch.long, device=past.device)
        logits = self.model(torch.cat((bos, targets[:, :-1]), dim=1), context)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    def reset(self):
        self._past_buffer = None
        self._past_valid_buffer = None
        self._pending_execution_steps = None

    @torch.no_grad()
    def predict_action(self, obs_dict, use_k_tokens=None, temperature=None, topk=None,
                       past_actions=None, past_action_valid=None):
        if (past_actions is None) != (past_action_valid is None):
            raise ValueError('Explicit past_actions and past_action_valid must be provided together')
        stateful = past_actions is None
        if stateful:
            if self._pending_execution_steps is not None:
                raise RuntimeError('Execution feedback is pending; call record_executed_actions or reset')
            batch = obs_dict[self.obs_ports[0]].shape[0]
            expected = (batch, self.past_n, self.action_dim)
            if self._past_buffer is None:
                self._past_buffer = torch.zeros(expected, device=self.device, dtype=self.dtype)
                self._past_valid_buffer = torch.zeros(expected[:2], device=self.device, dtype=torch.bool)
            elif tuple(self._past_buffer.shape) != expected:
                raise RuntimeError('Execution-history batch size changed; call reset first')
            self._past_buffer = self._past_buffer.to(device=self.device, dtype=self.dtype)
            self._past_valid_buffer = self._past_valid_buffer.to(device=self.device)
            past_actions, past_action_valid = self._past_buffer, self._past_valid_buffer
        n_tokens = self.max_seq_len if use_k_tokens is None else use_k_tokens
        if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or not 1 <= n_tokens <= self.max_seq_len:
            raise ValueError(f'use_k_tokens must be in [1,{self.max_seq_len}]')
        with self._rollout_mode(), self._clean_autocast_cache():
            prediction = self._generate_actions(
                obs_dict, past_actions, past_action_valid, n_tokens,
                self.temperature if temperature is None else temperature,
                self.topk if topk is None else topk,
            )
        action = prediction[:, :self.n_action_steps]
        if stateful:
            self._pending_execution_steps = action.shape[1]
        return {'action': action, 'action_pred': prediction}

    def record_executed_actions(self, actions, executed_lengths=None):
        if self._pending_execution_steps is None or self._past_buffer is None:
            raise RuntimeError('No prediction is awaiting execution feedback')
        commands = torch.as_tensor(actions, device=self._past_buffer.device)
        batch = self._past_buffer.shape[0]
        if commands.ndim != 3 or commands.shape[0] != batch or commands.shape[2] != self.action_dim:
            raise ValueError('actions must have shape (batch,steps,action_dim)')
        if commands.dtype == torch.bool or commands.is_complex():
            raise ValueError('Executed commands must be real numbers')
        steps = commands.shape[1]
        if steps > self._pending_execution_steps:
            raise ValueError('Acknowledged chunk exceeds the predicted execution horizon')
        lengths = torch.full((batch,), steps, device=commands.device, dtype=torch.long) if executed_lengths is None else torch.as_tensor(executed_lengths, device=commands.device)
        if lengths.shape != (batch,) or lengths.dtype == torch.bool or lengths.is_complex():
            raise ValueError('executed_lengths must be an integer vector of shape (batch,)')
        if lengths.is_floating_point() and (not torch.isfinite(lengths).all() or not torch.equal(lengths, lengths.floor())):
            raise ValueError('executed_lengths must contain finite integer counts')
        if (lengths < 0).any() or (lengths > steps).any():
            raise ValueError('executed_lengths must lie within the supplied chunk')
        executed = torch.arange(steps, device=commands.device)[None, :] < lengths[:, None]
        if not torch.isfinite(commands[executed]).all():
            raise ValueError('Executed commands must be finite')
        with torch.inference_mode(False), torch.no_grad():
            history = self._past_buffer.detach().clone()
            valid = self._past_valid_buffer.detach().clone()
            for i, count in enumerate(lengths.long().cpu().tolist()):
                if count:
                    history[i] = torch.cat((history[i], commands[i, :count].to(history.dtype)), dim=0)[-self.past_n:]
                    valid[i] = torch.cat((valid[i], valid.new_ones(count)))[-self.past_n:]
        self._past_buffer, self._past_valid_buffer = history, valid
        self._pending_execution_steps = None

    def get_optimizer(self, policy_lr=5e-5, obs_enc_lr=1e-4, weight_decay=0.01, betas=(0.9, 0.95)):
        groups, seen = {}, set()
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            visual = name.startswith('obs_encoder.') and not name.startswith('obs_encoder.state_projection.')
            lr = obs_enc_lr if visual else policy_lr
            decay = weight_decay if parameter.ndim >= 2 else 0.
            groups.setdefault((lr, decay), []).append(parameter)
        return torch.optim.AdamW([{'params': params, 'lr': lr, 'weight_decay': decay}
                                  for (lr, decay), params in groups.items()], betas=tuple(betas))

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities

    def get_observation_ports(self):
        return list(self.obs_ports)

    def get_policy_name(self):
        return f'{self.variant}_resnet18_{self.task}'

    def create_dummy_observation(self, batch_size=1, device=None):
        device = self.device if device is None else device
        obs = {}
        for key in self.obs_ports:
            shape = (batch_size, self.n_obs_steps, *self.obs_key_shapes[key])
            obs[key] = torch.zeros(shape, device=device,
                                   dtype=torch.uint8 if key in self.obs_encoder.rgb_ports else self.dtype)
            if key.endswith('_quat'):
                obs[key][..., 3] = 1
            elif key.endswith('_rot6d'):
                obs[key][...] = obs[key].new_tensor([1, 0, 0, 0, 1, 0])
        return obs

    def parameter_counts(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {'total': trainable + frozen, 'trainable': trainable, 'frozen': frozen}

    def export_config(self):
        if not self._tokenizer_config:
            raise ValueError('Export requires tokenizer_config for injected tokenizers')
        return dict(_target_=f'{type(self).__module__}.{type(self).__name__}', _recursive_=False,
                    **copy.deepcopy(self._construction), construction_mode='restore',
                    obs_encoder_config=self.obs_encoder.export_config(),
                    tokenizer_config=copy.deepcopy(self._tokenizer_config),
                    tokenizer_metadata=copy.deepcopy(self._tokenizer_metadata))
