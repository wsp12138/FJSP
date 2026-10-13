"""Run the Ours joint matching policy on user-provided checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .algorithms import OursConfig
from .runtime import check_actions, event_close_step
from .scenarios import SCENARIOS, make_environment
from .train import make_agent


def evaluate(scenario: str, checkpoint: Path, seed: int, episodes: int,
             config: OursConfig, instance_path: str | None = None) -> list[dict]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    torch.set_num_threads(1)
    env = make_environment(scenario, seed, instance_path)
    agent = make_agent(env, config)
    agent.load(checkpoint)
    rows = []
    for offset in range(episodes):
        obs, state, mask = env.reset(seed=seed + offset)
        done = False
        total_reward = 0.0
        decisions = 0
        while not done:
            policy_obs, policy_state = agent.prepare_policy_inputs(obs, state)
            actions, _, _ = agent.act(policy_obs, policy_state, mask, deterministic=True)
            check_actions(actions, mask)
            obs, state, reward, _, done, info, _ = event_close_step(env, actions)
            mask = env.action_masks()
            total_reward += float(reward[0])
            decisions += 1
        metrics = env.metrics()
        if metrics["completed_operations"] != metrics["total_operations"]:
            raise RuntimeError("Inference ended before all operations completed")
        rows.append({"scenario": scenario, "seed": seed + offset, "decisions": decisions,
                     "total_tardiness": float(metrics["total_tardiness"]),
                     "makespan": float(metrics["makespan"]), "episode_return": total_reward,
                     "complete": True, "invalid_actions": int(info.get("invalid_actions", 0))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="S20L")
    parser.add_argument("--seed", type=int, default=3600000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--instance", help="External .fjs or JSON file for Mk08/Mk09")
    parser.add_argument("--output", type=Path, help="Optional JSON output; otherwise prints to terminal")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    values = saved.get("config", {})
    if isinstance(values, OursConfig):
        config = values
    elif isinstance(values, dict):
        config = OursConfig(**values)
    else:
        raise ValueError("Checkpoint must contain the Ours configuration")
    rows = evaluate(args.scenario, args.checkpoint, args.seed, args.episodes, config, args.instance)
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
