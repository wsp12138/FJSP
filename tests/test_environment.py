"""Verify pure-tardiness rewards over a complete natural episode."""

import numpy as np
import pytest

from ours_fjsp.runtime import event_close_step
from ours_fjsp.scenarios import make_environment


def test_complete_natural_episode_reward_equals_negative_total_tardiness():
    env = make_environment("S20L", seed=31)
    env.reset(seed=31)
    active_rewards = {
        name: weight for name, weight in env.config.reward_weights.items() if weight != 0
    }
    assert active_rewards == {"delta_tardiness": 1.0}
    initial_tardiness = float(env.metrics()["total_tardiness"])
    episode_return = 0.0
    for _ in range(env.config.max_steps):
        mask = env.action_masks()
        actions = np.zeros(env.num_agents, dtype=np.int64)
        chosen = set()
        # An independent dispatcher keeps this environment test separate
        # from the learned actor and exact-policy implementation.
        for machine in range(env.num_agents):
            candidates = [int(action) for action in np.flatnonzero(mask[machine, 1:]) + 1
                          if int(action) not in chosen]
            if candidates:
                actions[machine] = candidates[0]
                chosen.add(candidates[0])
        _, _, reward, _, done, info, _ = event_close_step(env, actions)
        assert info.get("invalid_actions", 0) == 0
        assert np.isfinite(reward).all()
        episode_return += float(np.mean(reward))
        if done:
            break
    else:
        pytest.fail("The natural episode did not finish within the event limit")

    metrics = env.metrics()
    assert metrics["completed_operations"] == metrics["total_operations"]
    assert env.step_count < env.config.max_steps
    assert episode_return == pytest.approx(
        initial_tardiness - float(metrics["total_tardiness"]), abs=1e-3
    )
