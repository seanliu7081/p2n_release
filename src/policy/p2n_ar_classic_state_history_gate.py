"""Independent, observation-conditioned gate for state-history summary attention.

Only the appended summary tokens are gated. Original observation, raw action,
and command-difference conditions remain available. History encoding and execution acknowledgement are owned by this gated policy.
"""

import contextlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.model.autoregressive.transformer_cache_history_gate import (
    HistoryGatedAutoregressiveModel,
)
from src.model.state_action_history import StateActionHistoryEncoder
from src.policy.p2n_ar_classic import P2NARClassicPolicy

DEFAULT_STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


class P2NARClassicStateHistoryGatePolicy(P2NARClassicPolicy):
    """Add a shared scalar attention gate to the existing history summaries.

The gate observes pooled current observation features, pooled state/action
memory, and the fraction of valid history states. ``log(sigmoid(logit))`` is
added to summary-key attention scores in every AR layer, including cached
generation and offline self-past generation. It is a relative attention prior,
not the probability that the history is correct or a fraction of the action.

Modes ``open`` and ``closed`` are fixed diagnostic controls. An open gate
leaves history-summary attention unrestricted; closed masks only the appended
history summaries. The learned gate starts at ``history_gate_init``.
"""

    def __init__(self, *args, history_gate_mode="learned",
                 history_gate_hidden_dim=128, history_gate_init=0.9,
                 state_history_steps=8, state_history_keys=DEFAULT_STATE_KEYS,
                 history_embed_dim=128, history_n_heads=4, history_n_layers=2,
                 history_summary_tokens=4, history_dropout=0.1, **kwargs):
        if history_gate_mode not in {"learned", "open", "closed"}:
            raise ValueError("history_gate_mode must be learned, open, or closed")
        if (isinstance(history_gate_hidden_dim, bool)
                or not isinstance(history_gate_hidden_dim, int)
                or history_gate_hidden_dim < 1):
            raise ValueError("history_gate_hidden_dim must be a positive integer")
        if (isinstance(history_gate_init, bool)
                or not isinstance(history_gate_init, (int, float))
                or not math.isfinite(history_gate_init)
                or not 0 < history_gate_init < 1):
            raise ValueError("history_gate_init must be finite and strictly between 0 and 1")
        super().__init__(*args, **kwargs)
        if self.past_n < 3:
            raise ValueError("past_n must be at least 3 for command-difference features")
        if self.n_action_steps < 1:
            raise ValueError("n_action_steps must be positive")
        self._pending_execution_steps = None
        if state_history_steps != self.past_n + 1:
            raise ValueError("state_history_steps must equal past_n + 1")
        keys = tuple(state_history_keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("state_history_keys must be nonempty and unique")
        for key in keys:
            if key not in self.obs_key_shapes or len(self.obs_key_shapes[key]) != 1:
                raise ValueError(f"History key {key!r} must be a vector observation")
        self.state_history_steps = state_history_steps
        self.state_history_keys = keys
        self.history_encoder = StateActionHistoryEncoder(
            state_shapes={key: tuple(self.obs_key_shapes[key]) for key in keys},
            action_dim=self.action_dim, history_steps=state_history_steps,
            output_dim=self.obs_feature_dim, embed_dim=history_embed_dim,
            n_heads=history_n_heads, n_layers=history_n_layers,
            n_summary_tokens=history_summary_tokens, dropout=history_dropout,
        )
        # Fresh-policy architecture: retain all original conditions and append
        # fixed-size history memory. The tokenizer's architecture is unchanged.
        positions = self.model.cond_pos_emb
        self.model.cond_pos_emb = nn.Parameter(torch.cat((
            positions.detach(), positions.new_zeros(1, history_summary_tokens,
                                                   positions.shape[-1]),
        ), dim=1))
        history_params = sum(p.numel() for p in self.history_encoder.parameters())
        print(f"  state history: {state_history_steps} states, {self.past_n} commands, "
              f"{history_summary_tokens} summary tokens; {history_params / 1e6:.2f}M params; "
              f"total cond_len={self.model.cond_pos_emb.shape[1]}\n")

        self.history_gate_mode = history_gate_mode
        self.history_gate_init = float(history_gate_init)
        self.history_summary_tokens = self.history_encoder.n_summary_tokens
        self.model = HistoryGatedAutoregressiveModel.from_base(
            self.model, history_token_count=self.history_summary_tokens,
        )
        gate_input_dim = 2 * self.obs_feature_dim + 1
        self.history_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, history_gate_hidden_dim),
            nn.GELU(),
            nn.Linear(history_gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.history_gate[-1].weight)
        nn.init.constant_(self.history_gate[-1].bias,
                          math.log(self.history_gate_init / (1 - self.history_gate_init)))
        # Keep identical checkpoint keys for all modes, without unused trainable
        # gate parameters in the fixed controls under DDP.
        self.history_gate.requires_grad_(history_gate_mode == "learned")
        self.register_buffer("_history_gate_schema_version", torch.tensor(1, dtype=torch.long))
        self._last_history_gate = None

    def get_policy_name(self):
        return "p2n_ar_classic_state_history_gate_" + "|".join(
            modality for modality in self.modalities if modality != "state"
        )

    def get_optimizer(self, policy_lr, obs_enc_lr, weight_decay, betas):
        optimizer = super().get_optimizer(policy_lr, obs_enc_lr, weight_decay, betas)
        params = [p for p in self.history_encoder.parameters() if p.requires_grad]
        for decay, group in ((weight_decay, [p for p in params if p.ndim >= 2]),
                             (0.0, [p for p in params if p.ndim < 2])):
            optimizer.add_param_group({"params": group, "lr": policy_lr,
                                       "weight_decay": decay})
        params = [p for p in self.history_gate.parameters() if p.requires_grad]
        for decay, group in ((weight_decay, [p for p in params if p.ndim >= 2]),
                             (0.0, [p for p in params if p.ndim < 2])):
            if group:
                optimizer.add_param_group({"params": group, "lr": policy_lr,
                                           "weight_decay": decay})
        return optimizer


    def _condition_and_gate(self, obs, past_actions):
        # Validate history, mask padding, and compute observation
        # and history features exactly once. Original observations occupy the
        # first n_obs_steps positions; appended summaries occupy the last ones.
        cond = self._condition(obs, past_actions)
        batch_size = cond.shape[0]
        if self.history_gate_mode == "open":
            log_gate = None  # Also retains the original SDPA path when open.
            gate_value = cond.new_ones(batch_size, 1)
        elif self.history_gate_mode == "closed":
            log_gate = cond.new_full((batch_size, 1), -float("inf"))
            gate_value = cond.new_zeros(batch_size, 1)
        else:
            observed = cond[:, :self.n_obs_steps].mean(dim=1)
            history = cond[:, -self.history_summary_tokens:].mean(dim=1)
            valid_fraction = obs["state_history_valid"].to(
                device=cond.device, dtype=cond.dtype,
            ).mean(dim=1, keepdim=True)
            inputs = torch.cat((observed, history, valid_fraction), dim=-1)
            logits = self.history_gate(inputs)
            # Stable for large negative logits, including mixed precision;
            # unlike log(sigmoid(logits)), this does not underflow to -inf.
            log_gate = F.logsigmoid(logits.float())
            gate_value = log_gate.exp()
        self._last_history_gate = gate_value.detach()
        return cond, log_gate

    def get_history_gate_metrics(self):
        """Inspect the last call; synchronization occurs only on explicit use.

        The unchanged training workspace does not automatically log these
        diagnostics. No graph or persistent checkpoint statistics are retained.
        """
        if self._last_history_gate is None:
            return {}
        gate = self._last_history_gate.float()
        return {"history_gate/mean": gate.mean().item(),
                "history_gate/min": gate.min().item(),
                "history_gate/max": gate.max().item()}

    @contextlib.contextmanager
    def _rollout_mode(self):
        was_training = self.history_gate.training
        history_was_training = self.history_encoder.training
        try:
            self.history_gate.eval()
            self.history_encoder.eval()
            with super()._rollout_mode():
                yield
        finally:
            self.history_gate.train(was_training)
            self.history_encoder.train(history_was_training)

    def _generate_actions(self, obs, past_actions, n_tokens, temperature, topk):
        cond, log_gate = self._condition_and_gate(obs, past_actions)
        tokens = torch.full((cond.shape[0], 1), self.bos_id, dtype=torch.long,
                            device=cond.device)
        tokens = self.model.generate(
            tokens, cond=cond, max_new_tokens=n_tokens,
            temperature=temperature, top_k=topk, bos_id=self.bos_id,
            history_log_gate=log_gate,
        )[:, 1:]
        with torch.inference_mode():
            return self.action_tokenizer.detokenize(tokens=tokens)

    def forward(self, batch, history_mode=None):
        with torch.no_grad():
            targets = self.action_tokenizer.tokenize(batch["action"])
        past = batch["past_action"]
        if history_mode is None:
            history_mode = ("expert" if not self.training and
                            self.self_past_schedule == "optimizer_step" else "configured")
        if history_mode == "generated":
            past = self._maybe_self_past(batch, past, probability=1.0)
        elif history_mode == "configured":
            past = self._maybe_self_past(batch, past)
        elif history_mode != "expert":
            raise ValueError("history_mode must be 'expert', 'generated', or 'configured'")
        cond, log_gate = self._condition_and_gate(batch["obs"], past)
        bos = torch.full((targets.shape[0], 1), self.bos_id, dtype=torch.long,
                         device=targets.device)
        logits = self.model(torch.cat((bos, targets[:, :-1]), dim=1),
                            cond=cond, history_log_gate=log_gate)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        if self.training and self.self_past_schedule == "legacy":
            self._train_step += 1
        return loss


    def get_observation_ports(self):
        return list(super().get_observation_ports()) + [
            "state_history__" + key for key in self.state_history_keys
        ] + ["state_history_valid"]

    def create_dummy_observation(self, batch_size=1, device=None):
        device = self.device if device is None else device
        obs = super().create_dummy_observation(batch_size=batch_size, device=device)
        obs = {key: value.to(dtype=self.dtype) if isinstance(value, torch.Tensor)
               and value.is_floating_point() else value for key, value in obs.items()}
        for key in self.state_history_keys:
            value = torch.zeros(batch_size, self.state_history_steps,
                                *self.obs_key_shapes[key], device=device, dtype=self.dtype)
            if key.endswith("_quat"):
                value[:, -1, 3] = 1
            obs["state_history__" + key] = value
        valid = torch.zeros(batch_size, self.state_history_steps, device=device,
                            dtype=torch.bool)
        valid[:, -1] = True
        obs["state_history_valid"] = valid
        return obs

    def _condition(self, obs, past_actions):
        required = ["state_history__" + key for key in self.state_history_keys]
        required.append("state_history_valid")
        missing = [key for key in required if key not in obs]
        if missing:
            raise KeyError(f"Missing state history ports {missing}; use "
                           "ZarrDatasetWithStateHistory / LiberoStateHistoryRunner")
        features = self.obs_encoder(obs)
        expected = (features.shape[0], self.past_n, self.action_dim)
        if tuple(past_actions.shape) != expected:
            raise ValueError(f"past_actions must have shape {expected}")
        validity = obs["state_history_valid"]
        # LiberoRunner casts all ndarray ports (including boolean metadata) to
        # the policy dtype. Accept its exact 0/1 representation, then restore bool.
        if validity.dtype != torch.bool:
            if not ((validity == 0) | (validity == 1)).all():
                raise ValueError("state_history_valid must contain only zero/one flags")
            validity = validity.to(dtype=torch.bool)
        original = self._build_condition(features, past_actions)
        memory = self.history_encoder(
            {key: obs["state_history__" + key] for key in self.state_history_keys},
            validity,
            self.action_normalizer["action"].normalize(past_actions),
            self.action_normalizer,
        )
        return torch.cat((original, memory), dim=1)

    def _generate_prev_past(self, batch):
        previous = batch["prev_past_action"]
        with self._rollout_mode(), self._clean_autocast_cache():
            with torch.inference_mode():
                prediction = self._generate_actions(
                    batch["prev_obs"], previous, self.max_seq_len,
                    self.self_past_temperature, self.self_past_topk,
                )
                generated = torch.cat((previous, prediction[:, :self.n_action_steps]),
                                      dim=1)[:, -self.past_n:]
        with torch.inference_mode(False):
            return generated.detach().clone().to(dtype=previous.dtype)

    def predict_action(self, obs_dict, use_k_tokens=None, temperature=None,
                       topk=None, past_actions=None):
        stateful = past_actions is None
        if stateful and self._pending_execution_steps is not None:
            raise RuntimeError("Execution feedback is pending. Call record_executed_actions "
                               "or reset before another stateful prediction.")
        if use_k_tokens is None:
            use_k_tokens = self.max_seq_len
        if (isinstance(use_k_tokens, bool) or not isinstance(use_k_tokens, int)
                or use_k_tokens < 1):
            raise ValueError("use_k_tokens must be a positive integer")
        use_k_tokens = min(use_k_tokens, self.max_seq_len)
        if stateful:
            observation = next((value for value in obs_dict.values()
                                if isinstance(value, torch.Tensor)), None)
            if observation is None or observation.ndim < 1 or observation.shape[0] < 1:
                raise ValueError("A nonempty batched tensor observation is required")
            expected = (observation.shape[0], self.past_n, self.action_dim)
            if self._past_buffer is None:
                self._past_buffer = torch.zeros(expected, device=self.device, dtype=self.dtype)
            elif tuple(self._past_buffer.shape) != expected:
                raise RuntimeError("Execution-history batch size changed; call reset first")
            else:
                self._past_buffer = self._past_buffer.to(device=self.device, dtype=self.dtype)
            past_actions = self._past_buffer
        prediction = self._generate_actions(
            obs_dict, past_actions, use_k_tokens,
            self.temperature if temperature is None else temperature,
            self.topk if topk is None else topk,
        )
        action = prediction[:, :self.n_action_steps]
        if stateful:
            self._pending_execution_steps = action.shape[1]
        return {"action": action, "action_pred": prediction}

    def reset(self):
        super().reset()
        self._pending_execution_steps = None

    def record_executed_actions(self, actions, executed_lengths=None):
        """Append only commands confirmed as executed, then permit prediction.

        Args:
            actions: Tensor/array of shape (B, T, action_dim), in the same raw
                command coordinates as the training dataset. Values may differ
                from predictions when an execution layer modifies commands.
            executed_lengths: Integer prefix length per batch row, shape (B,).
                Omit only when every supplied command was actually executed.
                Zero leaves that row's history unchanged; unexecuted suffixes
                never contribute to history or its dynamic features.

        All inputs are validated before any history is changed. Acknowledgements
        describe a single proposed chunk, not arbitrary replay or future actions.
        """
        if self._pending_execution_steps is None or self._past_buffer is None:
            raise RuntimeError("No stateful prediction is awaiting execution feedback")

        commands = torch.as_tensor(actions, device=self.device)
        batch_size = self._past_buffer.shape[0]
        if (commands.ndim != 3 or commands.shape[0] != batch_size
                or commands.shape[2] != self.action_dim):
            raise ValueError("actions must have shape (batch_size, steps, action_dim)")
        steps = commands.shape[1]
        if steps > self._pending_execution_steps:
            raise ValueError("Acknowledged chunk exceeds the predicted execution horizon")
        if commands.is_complex() or commands.dtype == torch.bool:
            raise ValueError("Executed commands must contain real numeric values")
        commands = commands.detach().to(dtype=self._past_buffer.dtype)

        if executed_lengths is None:
            lengths = torch.full((batch_size,), steps, device=self.device, dtype=torch.long)
        else:
            lengths = torch.as_tensor(executed_lengths, device=self.device)
            if lengths.shape != (batch_size,) or lengths.dtype == torch.bool or lengths.is_complex():
                raise ValueError("executed_lengths must be an integer vector of shape (batch_size,)")
            if lengths.is_floating_point():
                if not torch.isfinite(lengths).all() or not torch.equal(lengths, lengths.floor()):
                    raise ValueError("executed_lengths must contain finite integer counts")
            if (lengths < 0).any() or (lengths > steps).any():
                raise ValueError("executed_lengths must lie between zero and the supplied chunk length")
            lengths = lengths.to(dtype=torch.long)

        executed = torch.arange(steps, device=self.device)[None, :] < lengths[:, None]
        if not torch.isfinite(commands).all(dim=-1)[executed].all():
            raise ValueError("Executed commands must be finite")

        # Clone outside inference mode so cached commands are ordinary detached
        # tensors even when acknowledgement is called inside inference_mode().
        with torch.inference_mode(False), torch.no_grad():
            history = self._past_buffer.detach().clone()
            for index, count in enumerate(lengths.cpu().tolist()):
                if count:
                    history[index] = torch.cat(
                        (history[index], commands[index, :count]), dim=0,
                    )[-self.past_n:]
        self._past_buffer = history
        self._pending_execution_steps = None

