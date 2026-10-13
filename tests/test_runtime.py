"""Exercise joint actions through the public agent and event runtime."""

import numpy as np
import pytest
import torch

from ours_fjsp.algorithms import OursAgent, OursConfig
from ours_fjsp.runtime import event_close_step
from ours_fjsp.scenarios import make_environment


def test_agent_actions_obey_capacity_masks_and_replay_the_joint_probability():
    torch.manual_seed(19)
    env = make_environment("S20L", seed=19)
    agent = OursAgent(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        num_agents=env.num_agents,
        max_jobs=env.max_jobs,
        config=OursConfig(),
    )
    obs, state, mask = env.reset(seed=19)
    for deterministic in (False, True, False, True):
        policy_obs, policy_state = agent.prepare_policy_inputs(obs, state)
        actions, log_probability, value = agent.act(
            policy_obs, policy_state, mask, deterministic=deterministic
        )
        assert actions.shape == (env.num_agents,)
        assert np.all((actions >= 0) & (actions < env.action_dim))
        assert all(mask[machine, action] for machine, action in enumerate(actions))
        selected_operations = actions[actions > 0]
        assert len(selected_operations) == len(set(selected_operations.tolist()))
        assert np.isfinite(log_probability) and np.isfinite(value)

        tokens = agent.last_pair_tokens.copy()
        selected_tokens = tokens[tokens >= 0]
        assert len(selected_tokens) == len(selected_operations)
        owners = selected_tokens // env.action_dim
        operations = selected_tokens % env.action_dim
        assert len(owners) == len(set(owners.tolist()))
        assert len(operations) == len(set(operations.tolist()))
        assert all(actions[owner] == operation
                   for owner, operation in zip(owners, operations))
        replayed, _ = agent.replay_log_probability(policy_obs, mask, tokens)
        assert replayed == pytest.approx(log_probability, abs=1e-6)

        obs, state, _, _, done, info, _ = event_close_step(env, actions)
        assert info.get("invalid_actions", 0) == 0
        if done:
            break
        mask = env.action_masks()
