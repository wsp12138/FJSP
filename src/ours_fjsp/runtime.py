"""Event closing and validity checks shared by training and inference."""
from __future__ import annotations

import numpy as np


def event_close_step(env, actions: np.ndarray):
    """Advance to the next observable event after a joint assignment."""
    previous_time = float(env.time)
    obs, state, rewards, masks, done, info = env.step(actions)
    assigned = int(info.get("assigned", 0))
    scalar = float(np.mean(rewards))
    objective = np.asarray([info.get("reward_components", {}).get("delta_tardiness", 0.0)])
    closes = 0
    if not done and np.count_nonzero(actions) and env.time <= previous_time + 1e-9:
        obs, state, next_rewards, masks, done, next_info = env.step(np.zeros(env.num_agents, dtype=np.int64))
        scalar += float(np.mean(next_rewards))
        objective += np.asarray([next_info.get("reward_components", {}).get("delta_tardiness", 0.0)])
        info = dict(next_info)
        info["assigned"] = assigned + int(next_info.get("assigned", 0))
        closes = 1
        if not done and env.time <= previous_time + 1e-9:
            raise RuntimeError("Event closing did not advance time")
    return obs, state, np.asarray([scalar]), objective, done, info, closes


def check_actions(actions: np.ndarray, masks: np.ndarray) -> None:
    """Check physical feasibility and one-operation-per-machine constraints."""
    if actions.shape != (masks.shape[0],):
        raise RuntimeError("Joint action has the wrong machine count")
    assigned = [int(a) for a in actions if a > 0]
    if len(assigned) != len(set(assigned)):
        raise RuntimeError("An operation was assigned to multiple machines")
    for machine, action in enumerate(actions):
        if int(action) < 0 or int(action) >= masks.shape[1] or (action > 0 and not masks[machine, action]):
            raise RuntimeError("Joint action violates the physical action mask")
    if np.any(masks[:, 1:]) and not assigned:
        raise RuntimeError("A feasible joint action cannot be entirely empty")
