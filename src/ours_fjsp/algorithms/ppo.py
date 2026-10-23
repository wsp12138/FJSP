from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch import nn

from .networks import BipartiteSetActor, ValueCritic
from .initialization import INITIALIZATION_PROFILE, initialize_paper_networks
from .policy import SetMatchingPolicy


@dataclass(frozen=True)
class OursConfig:
    """The complete method: graph messages, learned waiting and partial matchings."""

    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 2
    minibatch_size: int = 64
    hidden_dim: int = 64
    critic_value_scale: float = 1000.0
    normalize_returns: bool = True
    global_feature_dim: int = 6
    machine_feature_dim: int = 5
    op_feature_dim: int = 15
    device: str = 'cpu'

    allow_partial: ClassVar[bool] = True
    graph_messages: ClassVar[bool] = True
    wait_scores: ClassVar[bool] = True
    status_enabled: ClassVar[bool] = True
    entropy_coef: ClassVar[float] = 0.0
    attention_heads: ClassVar[int] = 0
    status_op_feature_index: ClassVar[int] = 14
    tie_priority_machine_feature_index: ClassVar[int] = 4
    initialization_profile: ClassVar[str] = INITIALIZATION_PROFILE

    def __post_init__(self):
        if self.gamma != 1.0:
            raise ValueError('Pure total-tardiness training requires gamma=1')
        if (self.global_feature_dim,self.machine_feature_dim,self.op_feature_dim)!=(6,5,15):
            raise ValueError('Ours requires the causal 6/5/15 observation profile')
        if not (self.learning_rate>0 and 0<=self.gae_lambda<=1 and self.clip_coef>0 and self.value_coef>=0 and self.max_grad_norm>0 and self.critic_value_scale>0):
            raise ValueError('Invalid PPO hyperparameters')
        if min(self.update_epochs,self.minibatch_size,self.hidden_dim)<1:
            raise ValueError('Network and minibatch sizes must be positive')


class RolloutBuffer:
    """Joint decisions, with latent pair tokens retained for exact PPO replay."""

    def __init__(self):
        self.obs=[]
        self.states=[]
        self.masks=[]
        self.actions=[]
        self.joint_log_probs=[]
        self.values=[]
        self.rewards=[]
        self.dones=[]

    def add(self, *, obs, state, mask, actions, joint_log_prob, value, reward, done,
            objective_reward=None, machine_target=None, job_target=None):
        # Compatibility inputs have zero loss weight in the complete pure-TT
        # method. They are accepted without introducing auxiliary objectives.
        del objective_reward, machine_target, job_target
        self.obs.append(np.asarray(obs,dtype=np.float32).copy())
        self.states.append(np.asarray(state,dtype=np.float32).copy())
        self.masks.append(np.asarray(mask,dtype=bool).copy())
        self.actions.append(np.asarray(actions,dtype=np.int64).copy())
        self.joint_log_probs.append(float(joint_log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.rewards)


class _CausalInputsAndPolicyAPI:

    def _causal_actor_observation(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Erase unarrived operations without conflating busy and repair."""
        if obs.ndim not in {2, 3}:
            raise ValueError('Ours actor observation must have rank 2 or 3')
        squeeze = obs.ndim == 2
        obs_b = obs.unsqueeze(0) if squeeze else obs
        result = obs_b.clone()
        batch = result.shape[0]
        g = int(self.config.global_feature_dim)
        md = int(self.config.machine_feature_dim)
        od = int(self.config.op_feature_dim)
        machine_start = g + self.num_agents
        operation_start = machine_start + self.num_agents * md
        operation_features = result[:, :, operation_start:].reshape(batch, self.num_agents, self.action_dim - 1, od)
        current_time = result[:, 0, 0]
        arrived = (operation_features[:, :, :, 0] > 0.5) & (operation_features[:, :, :, 8] <= current_time[:, None, None] + 1e-07)
        operation_features = torch.where(arrived.unsqueeze(-1), operation_features, torch.zeros_like(operation_features))
        status_channel = int(self.config.status_op_feature_index)
        if status_channel < od:
            operation_features[:, :, :, status_channel] = 0.0
        if g > 3:
            result[:, :, 3] = 0.0
        result[:, :, operation_start:] = operation_features.reshape(batch, self.num_agents, (self.action_dim - 1) * od)
        return result.squeeze(0) if squeeze else result

    def _causal_critic_state(self, state: torch.Tensor) -> torch.Tensor:
        """Keep arrived operations and the complete active-repair representation."""
        if state.ndim not in {1, 2}:
            raise ValueError('Ours critic state must have rank 1 or 2')
        squeeze = state.ndim == 1
        state_b = state.unsqueeze(0) if squeeze else state
        result = state_b.clone()
        g = int(self.config.global_feature_dim)
        md = int(self.config.machine_feature_dim)
        od = int(self.config.op_feature_dim)
        machine_end = g + self.num_agents * md
        operation_width = od + self.num_agents
        machines = result[:, g:machine_end].reshape(-1, self.num_agents, md).clone()
        operations = result[:, machine_end:].reshape(-1, self.action_dim - 1, operation_width)
        current_time = result[:, 0]
        arrived = (operations[:, :, 0] > 0.5) & (operations[:, :, 8] <= current_time[:, None] + 1e-07)
        operations = torch.where(arrived.unsqueeze(-1), operations, torch.zeros_like(operations))
        status_channel = int(self.config.status_op_feature_index)
        if status_channel < od:
            operations[:, :, status_channel] = 0.0
        if g > 3:
            result[:, 3] = 0.0
        machines[:, :, 0] = 0.0
        machines[:, :, int(self.config.tie_priority_machine_feature_index)] = 0.0
        result[:, g:machine_end] = machines.reshape(-1, self.num_agents * md)
        result[:, machine_end:] = operations.reshape(-1, (self.action_dim - 1) * operation_width)
        return result.squeeze(0) if squeeze else result

    def prepare_policy_inputs(self, obs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate the zeroed non-TT channel; Ours needs no random tie key."""
        obs_result = np.asarray(obs, dtype=np.float32).copy()
        state_result = np.asarray(state, dtype=np.float32).copy()
        if obs_result.shape != (self.num_agents, self.obs_dim):
            raise ValueError('Ours observation has the wrong shape')
        if state_result.shape != (self.state_dim,):
            raise ValueError('Ours state has the wrong shape')
        g = int(self.config.global_feature_dim)
        md = int(self.config.machine_feature_dim)
        machine_start = g + self.num_agents
        operation_start = machine_start + self.num_agents * md
        zero_index = int(self.config.tie_priority_machine_feature_index)
        obs_machines = obs_result[:, machine_start:operation_start].reshape(self.num_agents, self.num_agents, md)
        state_machines = state_result[g:g + self.num_agents * md].reshape(self.num_agents, md)
        if bool(np.any(np.abs(obs_machines[:, :, zero_index]) > 1e-08)) or bool(np.any(np.abs(state_machines[:, zero_index]) > 1e-08)):
            raise ValueError('Ours requires the tardiness-only zero machine channel; it does not consume energy or an exogenous tie priority')
        return (obs_result, state_result)

    def _residual_actor_observation(self, causal_obs: torch.Tensor, remaining_machines: torch.Tensor, remaining_operations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Delete consumed endpoints from the graph encoded by the actor."""
        if causal_obs.shape != (self.num_agents, self.obs_dim):
            raise ValueError('Ours residual encoder expects one joint observation')
        result = causal_obs.clone()
        g = int(self.config.global_feature_dim)
        md = int(self.config.machine_feature_dim)
        od = int(self.config.op_feature_dim)
        machine_start = g + self.num_agents
        operation_start = machine_start + self.num_agents * md
        machines = result[:, machine_start:operation_start].reshape(self.num_agents, self.num_agents, md)
        machines[:, ~remaining_machines, :] = 0.0
        result[:, machine_start:operation_start] = machines.reshape(self.num_agents, self.num_agents * md)
        operations = result[:, operation_start:].reshape(self.num_agents, self.action_dim - 1, od)
        technological = operations[:, :, 4] > 0.5
        context = technological & remaining_machines[:, None] & remaining_operations[1:][None, :]
        operations = torch.where(remaining_operations[1:][None, :, None], operations, torch.zeros_like(operations))
        operations[:, :, 4] = context.to(operations.dtype)
        operations[:, :, 5] = torch.where(context, operations[:, :, 5], torch.zeros_like(operations[:, :, 5]))
        result[:, operation_start:] = operations.reshape(self.num_agents, (self.action_dim - 1) * od)
        return (result, context)

    def _actor_logits_matrix(self, obs: torch.Tensor, mask: torch.Tensor, *, include_wait: bool=False) -> torch.Tensor:
        causal = self._causal_actor_observation(obs, mask)
        remaining_m = torch.ones(self.num_agents, dtype=torch.bool, device=obs.device)
        remaining_o = torch.ones(self.action_dim, dtype=torch.bool, device=obs.device)
        remaining_o[0] = False
        (residual_obs, _context) = self._residual_actor_observation(causal, remaining_m, remaining_o)
        positive_mask = mask.clone()
        if not include_wait:
            positive_mask[:, 0] = False
        return self.actor.forward_joint(residual_obs, positive_mask)

    def edge_logits(self, obs: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
            logits = self._actor_logits_matrix(obs_t, mask_t)
        return logits.detach().cpu().numpy()

    def decode_pair_tokens(self, pair_tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(pair_tokens, dtype=np.int64)
        if tokens.shape != (self.num_agents,):
            raise ValueError('Ours pair-token trace must have shape (num_agents,)')
        actions = np.zeros(self.num_agents, dtype=np.int64)
        used_operations: set[int] = set()
        padding_started = False
        for code in tokens.tolist():
            if int(code) == -1:
                padding_started = True
                continue
            if padding_started:
                raise ValueError('Ours non-padding token follows padding')
            (machine, action) = divmod(int(code), self.action_dim)
            if not (0 <= machine < self.num_agents and 0 < action < self.action_dim):
                raise ValueError(f'Invalid Ours pair token: {code}')
            if actions[machine] != 0 or action in used_operations:
                raise ValueError('Ours pair-token trace reuses an endpoint')
            actions[machine] = action
            used_operations.add(action)
        return actions

    def replay_log_probability(self, obs: np.ndarray, action_masks: np.ndarray, pair_tokens: np.ndarray) -> tuple[float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
        tokens_t = torch.as_tensor(pair_tokens, dtype=torch.long, device=self.device)
        (_, log_prob, entropy) = self._distribution(obs_t, mask_t, pair_tokens=tokens_t, sample=False)
        return (float(log_prob.detach().item()), float(entropy.detach().item()))

    @torch.no_grad()
    def act(self, obs: np.ndarray, state: np.ndarray, action_masks: np.ndarray, deterministic: bool=False) -> tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
        (env_actions_t, log_probability, _entropy) = self._distribution(obs_t, mask_t, sample=not deterministic)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        causal_state = self._causal_critic_state(state_t).unsqueeze(0)
        value = self.critic(causal_state)
        env_actions = env_actions_t.detach().cpu().numpy().astype(np.int64)
        self.last_proposal_actions = env_actions.copy()
        if not np.array_equal(self.decode_pair_tokens(self.last_pair_tokens), env_actions):
            raise RuntimeError('Ours latent pair trace does not decode to its action')
        return (env_actions, float(log_probability.item()), float(value.item()))

    def _preupdate_policy_audit(self, buffer: object, next_value: float) -> dict[str, float]:
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)
        values = np.asarray(buffer.values + [next_value], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for index in reversed(range(len(rewards))):
            next_nonterminal = 1.0 - dones[index]
            delta = rewards[index] + float(self.config.gamma) * values[index + 1] * next_nonterminal - values[index]
            last_gae = delta + float(self.config.gamma) * float(self.config.gae_lambda) * next_nonterminal * last_gae
            advantages[index] = last_gae
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)
        obs = torch.as_tensor(np.asarray(buffer.obs), dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(np.asarray(buffer.masks), dtype=torch.bool, device=self.device)
        choices = torch.as_tensor(np.asarray(buffer.actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(buffer.joint_log_probs), dtype=torch.float32, device=self.device)
        replayed: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for index in range(len(rewards)):
            (_, log_prob, entropy) = self._distribution(obs[index], masks[index], pair_tokens=choices[index], sample=False)
            replayed.append(log_prob)
            entropies.append(entropy)
        new_log_probs = torch.stack(replayed)
        ratio = torch.exp(new_log_probs - old_log_probs)
        advantage_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        policy_loss = -(ratio * advantage_t).mean()
        parameters = [parameter for parameter in self.actor.parameters() if parameter.requires_grad]
        gradients = torch.autograd.grad(policy_loss, parameters, allow_unused=True)
        squared_norm = sum((float(torch.sum(gradient.detach().square()).item()) for gradient in gradients if gradient is not None))
        replay_error = torch.max(torch.abs(new_log_probs.detach() - old_log_probs))
        return {'policy_grad_norm_preupdate': float(np.sqrt(squared_norm)), 'policy_loss_preupdate': float(policy_loss.detach().item()), 'replay_log_prob_max_abs_error': float(replay_error.item()), 'replay_ratio_max_abs_deviation': float(torch.max(torch.abs(ratio.detach() - 1.0)).item()), 'latent_entropy_preupdate': float(torch.stack(entropies).mean().detach().item())}


class OursAgent(_CausalInputsAndPolicyAPI):
    """PPO over exact unordered joint partial matchings.

    Only physically feasible edges define support. Graph scores and learned
    machine waiting scores define the joint Gibbs distribution. Training
    replays the same latent matching probability used when sampling.
    """

    def __init__(self, *, obs_dim: int, state_dim: int, action_dim: int,
                 num_agents: int, max_jobs: int, config: OursConfig | None = None):
        self.config = config or OursConfig()
        cfg = self.config
        requested = cfg.device
        self.device = torch.device('cuda' if requested == 'auto' and torch.cuda.is_available()
                                   else 'cpu' if requested == 'auto' else requested)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)
        self.max_jobs = int(max_jobs)
        if not 1 <= self.num_agents <= 16 or self.action_dim < 2:
            raise ValueError('Exact partial matching requires 1–16 machines and an operation column')
        expected_state = cfg.global_feature_dim + self.num_agents * cfg.machine_feature_dim \
            + (self.action_dim - 1) * (cfg.op_feature_dim + self.num_agents)
        if self.state_dim != expected_state:
            raise ValueError(f'Expected state_dim={expected_state}, got {state_dim}')
        self.actor, self.critic = initialize_paper_networks(
            obs_dim=self.obs_dim, state_dim=self.state_dim, action_dim=self.action_dim,
            num_agents=self.num_agents, max_jobs=self.max_jobs,
            hidden_dim=cfg.hidden_dim, global_feature_dim=cfg.global_feature_dim,
            machine_feature_dim=cfg.machine_feature_dim, op_feature_dim=cfg.op_feature_dim,
            value_scale=cfg.critic_value_scale,
        )
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.learning_rate)
        self.last_pair_tokens = np.full(self.num_agents, -1, dtype=np.int64)
        self.last_owner_choices = self.last_pair_tokens.copy()
        self.last_cardinality_probabilities = np.array([1.0], dtype=np.float32)
        self.last_matching_size = 0
        self.last_assigned_cardinality = 0
        self.last_max_cardinality = 0
        self.posthoc_repair_calls = 0

    def _distribution(self, obs: torch.Tensor, mask: torch.Tensor,
                      pair_tokens: torch.Tensor | None = None, sample: bool = False):
        if obs.shape != (self.num_agents, self.obs_dim):
            raise ValueError('One complete joint observation is required')
        if mask.shape != (self.num_agents, self.action_dim):
            raise ValueError('Physical action mask has the wrong shape')
        joint_logits = self._actor_logits_matrix(obs, mask, include_wait=True)
        # The learned waiting branch supplies a machine-specific opportunity
        # cost; all feasible nonempty partial matchings remain in support.
        logits = joint_logits[:, 1:] - joint_logits[:, :1]
        distribution = SetMatchingPolicy(logits, mask[:, 1:].clone())
        if pair_tokens is not None:
            decoded = self.decode_pair_tokens(pair_tokens.detach().cpu().numpy().astype(np.int64))
            operations = tuple(int(action - 1) if action > 0 else -1 for action in decoded)
        elif sample:
            operations = distribution.sample().operations_by_machine
        else:
            operations = distribution.mode()
        if pair_tokens is None:
            with torch.no_grad():
                self.last_cardinality_probabilities = torch.softmax(
                    distribution.log_partitions_by_cardinality(), dim=0).cpu().numpy()
        actions = torch.tensor([operation + 1 if operation >= 0 else 0 for operation in operations],
                               dtype=torch.long, device=self.device)
        log_probability = distribution.log_prob(operations)
        tokens = [machine * self.action_dim + action
                  for machine, action in enumerate(actions.tolist()) if action > 0]
        tokens.extend([-1] * (self.num_agents - len(tokens)))
        self.last_pair_tokens = np.asarray(tokens, dtype=np.int64)
        self.last_owner_choices = self.last_pair_tokens.copy()
        self.last_matching_size = len(tokens) - tokens.count(-1)
        self.last_assigned_cardinality = self.last_matching_size
        self.last_max_cardinality = distribution.cardinality
        return actions, log_probability, log_probability * 0.0

    @torch.no_grad()
    def value(self, state: np.ndarray) -> float:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return float(self.critic(self._causal_critic_state(state_t).unsqueeze(0)).item())

    def update(self, buffer: RolloutBuffer, next_value: float = 0.0) -> dict[str, float]:
        if not len(buffer):
            return {}
        audit = self._preupdate_policy_audit(buffer, next_value)
        before = [parameter.detach().clone() for parameter in self.actor.parameters()]
        cfg = self.config
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)
        values = np.asarray(buffer.values + [next_value], dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * values[t + 1] * next_nonterminal - values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        obs = torch.as_tensor(np.asarray(buffer.obs), dtype=torch.float32, device=self.device)
        states = self._causal_critic_state(torch.as_tensor(
            np.asarray(buffer.states), dtype=torch.float32, device=self.device))
        masks = torch.as_tensor(np.asarray(buffer.masks), dtype=torch.bool, device=self.device)
        tokens = torch.as_tensor(np.asarray(buffer.actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(buffer.joint_log_probs), dtype=torch.float32,
                                        device=self.device)
        raw_returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        if cfg.normalize_returns:
            return_mean = raw_returns_t.mean()
            return_std = raw_returns_t.std(unbiased=False).clamp_min(1e-6)
        else:
            return_mean = raw_returns_t.new_tensor(0.0)
            return_std = raw_returns_t.new_tensor(1.0)
        returns_t = (raw_returns_t - return_mean) / return_std
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        indices = np.arange(len(buffer))
        policy_losses, value_losses, raw_value_losses = [], [], []
        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), cfg.minibatch_size):
                mb = indices[start:start + cfg.minibatch_size]
                log_probs = torch.stack([
                    self._distribution(obs[i], masks[i], pair_tokens=tokens[i])[1] for i in mb])
                ratio = torch.exp(log_probs - old_log_probs[mb])
                unclipped = ratio * advantages_t[mb]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * advantages_t[mb]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_pred = self.critic(states[mb])
                normalized_value = (value_pred - return_mean) / return_std
                value_loss = 0.5 * (returns_t[mb] - normalized_value).pow(2).mean()
                raw_value_loss = 0.5 * (raw_returns_t[mb] - value_pred).pow(2).mean()
                self.optimizer.zero_grad(set_to_none=True)
                (policy_loss + cfg.value_coef * value_loss).backward()
                if cfg.critic_value_scale == 1.0:
                    nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                             cfg.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                raw_value_losses.append(float(raw_value_loss.item()))
        delta_squared = sum(float(torch.sum((new.detach() - old).square()).item())
                            for old, new in zip(before, self.actor.parameters()))
        parameter_squared = sum(float(torch.sum(old.square()).item()) for old in before)
        gradient_squared = sum(float(torch.sum(p.grad.detach().square()).item())
                               for p in self.actor.parameters() if p.grad is not None)
        return {
            'policy_loss': float(np.mean(policy_losses)),
            'value_loss': float(np.mean(value_losses)),
            'raw_value_loss': float(np.mean(raw_value_losses)),
            'mean_return': float(np.sum(rewards)),
            'mean_reward': float(np.mean(rewards)),
            'return_mean': float(np.mean(returns)),
            'return_std': float(np.std(returns)),
            'actor_parameter_delta_l2': float(np.sqrt(delta_squared)),
            'actor_parameter_relative_delta': float(np.sqrt(delta_squared) / max(1e-12, np.sqrt(parameter_squared))),
            'actor_grad_norm_last_minibatch': float(np.sqrt(gradient_squared)),
            **audit,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'format': 'ours_joint_matching_ppo_v1',
                    'initialization_profile': INITIALIZATION_PROFILE,
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(), 'config': asdict(self.config),
                    'obs_dim': self.obs_dim, 'state_dim': self.state_dim,
                    'action_dim': self.action_dim, 'num_agents': self.num_agents,
                    'max_jobs': self.max_jobs}, destination)

    def load(self, path: str | Path) -> None:
        data = torch.load(Path(path), map_location=self.device, weights_only=True)
        if data.get('format') != 'ours_joint_matching_ppo_v1':
            raise ValueError('Checkpoint is not an Ours source-package checkpoint')
        if data.get('initialization_profile', INITIALIZATION_PROFILE) != INITIALIZATION_PROFILE:
            raise ValueError('Checkpoint initialization profile does not match this agent')
        for field in ('obs_dim', 'state_dim', 'action_dim', 'num_agents', 'max_jobs'):
            if data[field] != getattr(self, field):
                raise ValueError(f'Checkpoint {field} does not match this environment')
        saved_config = {key: value for key, value in data['config'].items() if key != 'device'}
        current_config = {key: value for key, value in asdict(self.config).items() if key != 'device'}
        if saved_config != current_config:
            raise ValueError('Checkpoint configuration does not match this agent')
        self.actor.load_state_dict(data['actor'], strict=True)
        self.critic.load_state_dict(data['critic'], strict=True)
