"""Train the complete Ours policy from scratch."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from .algorithms import OursAgent, OursConfig, RolloutBuffer
from .runtime import check_actions, event_close_step
from .scenarios import SCENARIOS, make_environment


def make_agent(env, config: OursConfig) -> OursAgent:
    return OursAgent(obs_dim=env.obs_dim, state_dim=env.state_dim,
                     action_dim=env.action_dim, num_agents=env.num_agents,
                     max_jobs=env.max_jobs, config=config)


def train(scenario: str, seed: int, instance_seed: int, updates: int, rollout_steps: int,
          output: Path, config: OursConfig, instance_path: str | None = None) -> Path:
    if updates < 1 or rollout_steps < 1:
        raise ValueError("updates and rollout_steps must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    env = make_environment(scenario, instance_seed, instance_path)
    agent = make_agent(env, config)
    output.mkdir(parents=True, exist_ok=True)
    obs, state, mask = env.reset(seed=instance_seed)
    completed_episodes = 0
    episode_return = 0.0
    history = []
    for update in range(1, updates + 1):
        buffer = RolloutBuffer()
        reward_sum = 0.0
        episode_reward_sum = 0.0
        episode_count = 0
        for _ in range(rollout_steps):
            machine_target, job_target = env.bottleneck_targets(max_jobs=env.max_jobs)
            policy_obs, policy_state = agent.prepare_policy_inputs(obs, state)
            actions, logp, value = agent.act(policy_obs, policy_state, mask)
            check_actions(actions, mask)
            latent = agent.last_pair_tokens.copy()
            replay, _ = agent.replay_log_probability(policy_obs, mask, latent)
            if abs(logp - replay) > 1e-5:
                raise RuntimeError("Collected and replayed joint probabilities differ")
            next_obs, next_state, reward, objective, done, info, _ = event_close_step(env, actions)
            scalar = float(reward[0])
            buffer.add(obs=policy_obs, state=policy_state, mask=mask, actions=latent,
                       joint_log_prob=logp, value=value, reward=scalar,
                       objective_reward=objective, done=done,
                       machine_target=machine_target, job_target=job_target)
            reward_sum += scalar
            episode_return += scalar
            obs, state, mask = next_obs, next_state, env.action_masks()
            if done:
                metrics = env.metrics()
                if metrics["completed_operations"] != metrics["total_operations"]:
                    raise RuntimeError("Training episode ended before all operations completed")
                if abs(episode_return + float(metrics["total_tardiness"])) > 1e-2:
                    raise RuntimeError("Episode return does not equal negative total tardiness")
                episode_count += 1
                episode_reward_sum += episode_return
                completed_episodes += 1
                episode_return = 0.0
                obs, state, mask = env.reset(seed=instance_seed + completed_episodes)
        _, policy_state = agent.prepare_policy_inputs(obs, state)
        stats = agent.update(buffer, next_value=agent.value(policy_state))
        history.append({"update": update, "decisions": rollout_steps,
                        "decision_reward_sum": reward_sum,
                        "completed_episode_reward_sum": episode_reward_sum,
                        "completed_episode_count": episode_count, **stats})
        if update == 1 or update % 10 == 0 or update == updates:
            average = episode_reward_sum / episode_count if episode_count else None
            print(json.dumps({"update": update, "updates": updates, "episodes": completed_episodes,
                              "mean_episode_reward": average}, ensure_ascii=False), flush=True)
    checkpoint = output / "checkpoint.pt"
    agent.save(checkpoint)
    (output / "training.json").write_text(json.dumps({
        "scenario": scenario, "seed": seed, "instance_seed": instance_seed,
        "updates": updates, "rollout_steps": rollout_steps,
        "completed_training_episodes": completed_episodes,
        "algorithm": asdict(config), "environment": asdict(env.config),
        "updates_log": history,
    }, ensure_ascii=False, indent=2))
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional YAML configuration")
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--instance-seed", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--instance", help="External .fjs or JSON file for Mk08/Mk09")
    parser.add_argument("--output", type=Path, required=True, help="Directory for runtime outputs")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text()) if args.config else {}
    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a mapping")
    scenario = args.scenario or settings.get("scenario", "S20L")
    seed = args.seed if args.seed is not None else settings.get("seed", 2400001)
    instance_seed = args.instance_seed if args.instance_seed is not None else settings.get("instance_seed", SCENARIOS[scenario].first_instance_seed)
    config = OursConfig(**settings.get("algorithm", {}))
    checkpoint = train(scenario, int(seed), int(instance_seed),
                       args.updates if args.updates is not None else int(settings.get("updates", 300)),
                       args.rollout_steps if args.rollout_steps is not None else int(settings.get("rollout_steps", 512)),
                       args.output, config, args.instance or settings.get("instance"))
    print(f"Checkpoint saved to {checkpoint}")


if __name__ == "__main__":
    main()

