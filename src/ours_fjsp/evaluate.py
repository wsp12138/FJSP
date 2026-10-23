"""Evaluate Matching-MAPPO on the paper's common physical test instances."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import numpy as np

from .algorithms import OursConfig
from .runtime import check_actions, event_close_step, physical_instance_signatures
from .scenarios import SCENARIOS, evaluation_seed_start, make_environment
from .train import make_agent
from .protocol import checkpoint_config, file_sha256, validate_checkpoint_protocol


def evaluate(scenario: str, checkpoint: Path, seed: int, episodes: int,
             config: OursConfig, instance_path: str | None = None) -> list[dict]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    torch.set_num_threads(1)
    env = make_environment(scenario, seed, instance_path)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint_protocol(saved, scenario, env)
    agent = make_agent(env, config)
    agent.load(checkpoint)
    checkpoint_sha = file_sha256(checkpoint)
    rows = []
    for offset in range(episodes):
        obs, state, mask = env.reset(seed=seed + offset)
        signatures = physical_instance_signatures(env)
        done = False
        total_reward = 0.0
        decisions = 0
        invalid_actions = 0
        max_replay_error = 0.0
        while not done:
            policy_obs, policy_state = agent.prepare_policy_inputs(obs, state)
            actions, logp, _ = agent.act(policy_obs, policy_state, mask, deterministic=True)
            check_actions(actions, mask)
            replay, _ = agent.replay_log_probability(policy_obs, mask, agent.last_pair_tokens.copy())
            if not all(math.isfinite(number) for number in (logp, replay)):
                raise RuntimeError("Evaluation policy produced a nonfinite probability")
            max_replay_error = max(max_replay_error, abs(logp - replay))
            if max_replay_error > 1e-6:
                raise RuntimeError("Evaluation joint probability replay differs")
            obs, state, reward, _, done, info, _ = event_close_step(env, actions)
            mask = env.action_masks()
            total_reward += float(reward[0])
            if not math.isfinite(total_reward):
                raise RuntimeError("Evaluation environment produced a nonfinite reward")
            invalid_actions += int(info.get("invalid_actions", 0))
            decisions += 1
        metrics = env.metrics()
        complete = metrics["completed_operations"] == metrics["total_operations"]
        if invalid_actions:
            raise RuntimeError("Evaluation produced an invalid action")
        if complete and abs(total_reward + float(metrics["total_tardiness"])) > 1e-3:
            raise RuntimeError("Evaluation failed the pure-tardiness reward identity")
        flow_times = ([env.op_completion[env.job_last_op[j]] - job.arrival_time
                       for j, job in enumerate(env.instance.jobs)] if complete else None)
        rows.append({"scenario": scenario, "seed": seed + offset, "decisions": decisions,
                     "total_tardiness": float(metrics["total_tardiness"]) if complete else None,
                     "mean_flow_time": float(np.mean(flow_times)) if complete else None,
                     "makespan": float(metrics["makespan"]) if complete else None,
                     "episode_return": total_reward, "complete": bool(complete),
                     "truncated": not complete,
                     "completed_operations": int(metrics["completed_operations"]),
                     "total_operations": int(metrics["total_operations"]),
                     "operation_completion_rate": metrics["completed_operations"] / metrics["total_operations"],
                     "observed_total_tardiness_at_truncation": float(metrics["total_tardiness"]) if not complete else None,
                     "unfinished_penalty": 0.0, "invalid_actions": invalid_actions,
                     "max_replay_error": max_replay_error,
                     "checkpoint_sha256": checkpoint_sha,
                     "training_budget_classification": saved["paper_metadata"].get("budget", {}).get("classification", "unspecified"),
                     **signatures})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="S20L")
    parser.add_argument("--seed", type=int, help="Default: this scenario's paper test pool")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--instance", help="External .fjs or JSON file for Mk08/Mk09")
    parser.add_argument("--output", type=Path, help="Optional JSON output; otherwise prints to terminal")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint_config(saved)
    seed = evaluation_seed_start(args.scenario) if args.seed is None else args.seed
    rows = evaluate(args.scenario, args.checkpoint, seed, args.episodes, config, args.instance)
    text = json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
