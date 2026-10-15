"""Direct continuous-action flow with acknowledged Past2Next history.

No tokenizer, quantizer, or autoregressive decoder is constructed.
"""
from __future__ import annotations
import copy
import torch
from src.model.common.normalizer import LinearNormalizer
from src.policy.base_policy import BasePolicy
from src.policy.p2n_ar_modern_common import P2NARModernCommonPolicy, _plain, bool_mask

FLOW_DEFAULTS = dict(space='normalized_actions', ffn_type='gelu', ffn_hidden_dim=3072,
    self_qk_norm=False, cross_qk_norm='layernorm', qkv_bias=True, time_embed_dim=128,
    global_condition='time_and_step_only', pre_norm_modality=False,
    fm_fraction=0.75, ct_weight=1.0, fm_beta=[1.0, 1.5], fm_time_scale=0.999,
    ct_time_bins=10, teacher_dt_mode='same_relative_dt',
    loss_padding='supervise_edge_repeated_targets', solver='euler',
    inference_steps=8, self_past_steps=8, output_transform='action_unnormalize')


def validate_flow_config(value):
    value = _plain(value) or {}
    unknown = set(value) - set(FLOW_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown continuous_action_flow configuration fields: {sorted(unknown)}')
    config = {**copy.deepcopy(FLOW_DEFAULTS), **value}
    flexible = {'ffn_hidden_dim', 'time_embed_dim', 'inference_steps', 'self_past_steps'}
    for key in set(FLOW_DEFAULTS) - flexible:
        if config[key] != FLOW_DEFAULTS[key]:
            raise ValueError(f'This recipe requires flow.{key}={FLOW_DEFAULTS[key]!r}')
    for key in flexible:
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'flow.{key} must be a positive integer')
    return config

class P2NActionFlowCommonPolicy(BasePolicy):
    VARIANT = 'p2n_action_flow'
    policy_family = 'continuous_action_flow'
    supports_flow_loss_preparation = True
    supports_flow_sampling = True
    ARTIFACT_SCHEMA_VERSION = 1
    CONTEXT_SCHEMA_VERSION = 1
    supports_explicit_past_actions = True
    supports_explicit_past_action_valid = True
    supports_generated_history_validation = True
    requires_execution_acknowledgement = True
    requires_state_history = False
    supports_history_summary_gate = False

    # Reuse only codec-independent contracts, without invoking an AR constructor.
    _safe_past = P2NARModernCommonPolicy._safe_past
    self_past_step = P2NARModernCommonPolicy.self_past_step
    set_self_past_step = P2NARModernCommonPolicy.set_self_past_step
    on_optimizer_step = P2NARModernCommonPolicy.on_optimizer_step
    self_past_probability = P2NARModernCommonPolicy.self_past_probability
    _clean_autocast_cache = P2NARModernCommonPolicy._clean_autocast_cache
    record_executed_actions = P2NARModernCommonPolicy.record_executed_actions
    get_optimizer = P2NARModernCommonPolicy.get_optimizer
    get_observation_encoder = P2NARModernCommonPolicy.get_observation_encoder
    get_observation_modalities = P2NARModernCommonPolicy.get_observation_modalities
    get_observation_ports = P2NARModernCommonPolicy.get_observation_ports
    get_policy_name = P2NARModernCommonPolicy.get_policy_name
    create_dummy_observation = P2NARModernCommonPolicy.create_dummy_observation
    parameter_counts = P2NARModernCommonPolicy.parameter_counts



    def reset(self):
        self._past_buffer = None
        self._past_valid_buffer = None
        self._pending_execution_steps = None
        self._online_generator = None




    def set_normalizer(self, normalizer):
        if isinstance(normalizer, (list, tuple)):
            if len(normalizer) != 1:
                raise ValueError('Expected exactly one task normalizer')
            normalizer = normalizer[0]
        selected = LinearNormalizer()
        for key in ['action', *self.obs_encoder.state_ports]:
            if key not in normalizer.params_dict:
                raise KeyError(f'Missing training normalizer field: {key}')
            selected[key] = copy.deepcopy(normalizer[key])
        if any(not torch.isfinite(value).all() for value in selected.state_dict().values()):
            raise ValueError('Training normalizer must contain finite statistics')
        self.action_normalizer.load_state_dict(selected.state_dict(), strict=True)
        self.action_normalizer.to(self.device).requires_grad_(False)
        self._normalizer_ready.fill_(True)

    def _encode_target(self, actions):
        if not bool(self._normalizer_ready):
            raise RuntimeError('Set the training-set normalizer before using this policy')
        if actions.ndim != 3 or actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError('Action target must be [B,horizon,action_dim]')
        if not torch.isfinite(actions).all():
            raise ValueError('Action targets must be finite')
        with torch.autocast(device_type=actions.device.type, enabled=False):
            return self.action_normalizer['action'].normalize(actions.float()).float()

    def export_config(self):
        return dict(_target_=f'{type(self).__module__}.{type(self).__name__}', _recursive_=False,
            **copy.deepcopy(self._construction), construction_mode='restore',
            obs_encoder_config=self.obs_encoder.export_config(),
            normalizer_metadata=copy.deepcopy(self.normalizer_metadata))

    def normalized_observation(self, obs):
        normalized = dict(obs)
        for key in self.obs_encoder.state_ports:
            if not torch.isfinite(obs[key]).all():
                raise ValueError(f'Current observation {key!r} must be finite')
            normalized[key] = self.action_normalizer[key].normalize(obs[key].float())
        return normalized






    @torch.no_grad()
    def _maybe_self_past(self, batch, past, probability=None, generator=None):
        probability = self.self_past_probability() if probability is None else probability
        if not 0 <= probability <= 1:
            raise ValueError('Self-past probability must be in [0,1]')
        if probability == 0:
            return past.detach()
        required = ('prev_obs','prev_past_action','prev_past_action_valid','prev_window_valid')
        if any(key not in batch for key in required):
            raise KeyError('Self-past requires previous-window observations and validity metadata')
        windows = bool_mask(batch['prev_window_valid'].reshape(-1), (past.shape[0],),
                            'prev_window_valid', past.device)
        selected = windows.clone()
        if probability < 1:
            selected &= torch.rand(past.shape[0], device=past.device, generator=generator) < probability
        indices = selected.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return past.detach()
        generated = past.detach().clone()
        # Exclude invalid windows before gate encoding (their states can all be padding).
        with self._rollout_mode(), self._clean_autocast_cache():
            for index in indices.split(self.self_past_chunk_size):
                previous = batch['prev_past_action'][index]
                prediction = self._generate_actions(
                    {key: value[index] for key, value in batch['prev_obs'].items()},
                    previous, batch['prev_past_action_valid'][index],
                    num_flow_steps=self.flow['self_past_steps'], generator=generator)
                chunk = torch.cat((previous, prediction[:, :self.n_action_steps]), dim=1)[:, -self.past_n:]
                # no_grad rather than inference_mode keeps tensors safe for later autograd.
                generated[index] = chunk.detach().clone().to(past.dtype)
        valid = bool_mask(batch['past_action_valid'], past.shape[:2], 'past_action_valid', past.device)
        return torch.where((selected[:, None] & valid)[..., None], generated, past).detach()

    @torch.no_grad()
    def predict_action(self, obs_dict, past_actions=None, past_action_valid=None,
                       num_flow_steps=None, generator=None):
        if (past_actions is None) != (past_action_valid is None):
            raise ValueError('Explicit past_actions and bool past_action_valid must be provided together')
        stateful = past_actions is None
        if stateful and generator is None:
            if self._online_generator is None:
                self._online_generator = torch.Generator(device=self.device).manual_seed(self.online_seed)
            generator = self._online_generator
        if stateful:
            if self._pending_execution_steps is not None:
                raise RuntimeError('Execution feedback is pending; call record_executed_actions or reset')
            batch = obs_dict[self.obs_ports[0]].shape[0]
            expected = (batch, self.past_n, self.action_dim)
            if self._past_buffer is None:
                self._past_buffer = torch.zeros(expected, device=self.device, dtype=torch.float32)
                self._past_valid_buffer = torch.zeros(expected[:2], device=self.device, dtype=torch.bool)
            elif tuple(self._past_buffer.shape) != expected:
                raise RuntimeError('Execution-history batch size changed; call reset first')
            self._past_buffer = self._past_buffer.to(self.device)
            self._past_valid_buffer = self._past_valid_buffer.to(self.device)
            past_actions, past_action_valid = self._past_buffer, self._past_valid_buffer
        if past_action_valid.dtype != torch.bool:
            raise ValueError('past_action_valid must be bool')
        with self._rollout_mode(), self._clean_autocast_cache():
            prediction = self._generate_actions(obs_dict, past_actions, past_action_valid,
                num_flow_steps=num_flow_steps, generator=generator)
        action = prediction[:, :self.n_action_steps]
        if stateful:
            self._pending_execution_steps = action.shape[1]
        return {'action': action, 'action_pred': prediction}
