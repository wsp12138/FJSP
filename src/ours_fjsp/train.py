"""Train Matching-MAPPO with the paper's prefix-plus-minimum-episode budget."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from .algorithms import OursAgent, OursConfig, RolloutBuffer
from .runtime import check_actions, event_close_step
from .scenarios import SCENARIOS, make_environment, reference_instance_matches
from .protocol import (TRAIN_SEEDS, checkpoint_metadata, training_seed_range,
                       validate_environment)


@dataclass(frozen=True)
class TrainingBudget:
    """The fixed prefix is always completed before the natural-episode gate."""

    prefix_updates: int = 300
    target_episodes: int = 1200
    rollout_steps: int = 512
    smoke_updates: int | None = None

    def __post_init__(self):
        if min(self.prefix_updates, self.target_episodes, self.rollout_steps) < 1:
            raise ValueError("Training budgets must be positive")
        if self.smoke_updates is not None and self.smoke_updates < 1:
            raise ValueError("Smoke updates must be positive")

    def needs_update(self, state: dict) -> bool:
        if self.smoke_updates is not None:
            return state["updates"] < self.smoke_updates
        return (state["updates"] < self.prefix_updates or
                state["completed_training_episodes"] < self.target_episodes)

    def is_extension(self, state: dict) -> bool:
        return self.smoke_updates is None and state["updates"] >= self.prefix_updates


def make_agent(env, config: OursConfig) -> OursAgent:
    return OursAgent(obs_dim=env.obs_dim, state_dim=env.state_dim,
                     action_dim=env.action_dim, num_agents=env.num_agents,
                     max_jobs=env.max_jobs, config=config)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def output_lock(output: Path):
    """Keep two local training commands from writing the same output."""
    import fcntl
    with (output / ".training.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another training process owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def next_instance_seed(start: int, end: int | None, ended: int) -> int:
    return start + (ended if end is None else ended % (end - start + 1))


def initial_state(env, start: int) -> dict:
    obs, state, mask = env.reset(seed=start)
    return {"obs": obs, "state": state, "mask": mask, "episode_return": 0.0,
            "episode_decisions": 0, "updates": 0, "total_training_decisions": 0,
            "completed_training_episodes": 0, "ended_training_episodes": 0,
            "truncated_training_episodes": 0, "max_replay_error": 0.0,
            "updates_log": [], "episodes": []}


def collect_update(agent, env, state: dict, budget: TrainingBudget,
                   start: int, end: int | None) -> dict:
    """Collect a continuous rollout, shortening only an extension's final tail."""
    extension = budget.is_extension(state)
    buffer = RolloutBuffer()
    reward_sum, natural_reward_sum, natural_count = 0.0, 0.0, 0
    ended_count, truncated_count, truncated_reward_sum = 0, 0, 0.0
    target_terminal = False
    for _ in range(budget.rollout_steps):
        policy_obs, policy_state = agent.prepare_policy_inputs(state["obs"], state["state"])
        actions, logp, value = agent.act(policy_obs, policy_state, state["mask"], deterministic=False)
        check_actions(actions, state["mask"])
        latent = agent.last_pair_tokens.copy()
        replay, _ = agent.replay_log_probability(policy_obs, state["mask"], latent)
        if not all(math.isfinite(number) for number in (logp, replay, value)):
            raise RuntimeError("Training policy produced a nonfinite value")
        state["max_replay_error"] = max(state["max_replay_error"], abs(logp - replay))
        if abs(logp - replay) > 1e-6:
            raise RuntimeError("Collected and replayed joint probabilities differ")
        next_obs, next_state, reward, _, done, info, _ = event_close_step(env, actions)
        scalar = float(reward[0])
        if not math.isfinite(scalar):
            raise RuntimeError("Training environment produced a nonfinite reward")
        if info.get("invalid_actions", 0):
            raise RuntimeError("Environment reported an invalid training action")
        buffer.add(obs=policy_obs, state=policy_state, mask=state["mask"], actions=latent,
                   joint_log_prob=logp, value=value, reward=scalar, done=done)
        reward_sum += scalar
        state["episode_return"] += scalar
        state["episode_decisions"] += 1
        state["total_training_decisions"] += 1
        state["obs"], state["state"], state["mask"] = next_obs, next_state, env.action_masks()
        if not done:
            continue
        metrics = env.metrics()
        complete = metrics["completed_operations"] == metrics["total_operations"]
        if not complete and not extension:
            raise RuntimeError("Training prefix ended before all operations completed")
        observed_tt = float(info.get("total_tardiness", metrics["total_tardiness"]))
        if complete and abs(state["episode_return"] + observed_tt) > 1e-3:
            raise RuntimeError("Episode return does not equal negative total tardiness")
        state["ended_training_episodes"] += 1
        ended_count += 1
        if complete:
            state["completed_training_episodes"] += 1
            natural_count += 1
            natural_reward_sum += state["episode_return"]
        else:
            state["truncated_training_episodes"] += 1
            truncated_count += 1
            truncated_reward_sum += state["episode_return"]
        state["episodes"].append({
            "episode": state["ended_training_episodes"],
            "natural_episode": state["completed_training_episodes"] if complete else None,
            "instance_seed": next_instance_seed(start, end, state["ended_training_episodes"] - 1),
            "episode_return": state["episode_return"],
            "decisions": state["episode_decisions"], "complete": complete,
            "truncated": not complete, "total_tardiness": observed_tt if complete else None,
            "observed_total_tardiness_at_truncation": observed_tt if not complete else None,
            "unfinished_penalty": 0.0,
        })
        state["episode_return"], state["episode_decisions"] = 0.0, 0
        if extension and complete and state["completed_training_episodes"] == budget.target_episodes:
            target_terminal = True
            break
        ended = state["ended_training_episodes"]
        if not extension and end is not None and start + ended > end:
            raise RuntimeError("Training prefix instance seed partition exhausted")
        state["obs"], state["state"], state["mask"] = env.reset(
            seed=next_instance_seed(start, end, ended))
    _, policy_state = agent.prepare_policy_inputs(state["obs"], state["state"])
    next_value = 0.0 if target_terminal else agent.value(policy_state)
    stats = agent.update(buffer, next_value=next_value)
    if not all(math.isfinite(float(stats[key])) for key in ("policy_loss", "value_loss")):
        raise RuntimeError("Nonfinite PPO loss")
    if not all(torch.isfinite(parameter).all().item()
               for module in (agent.actor, agent.critic) for parameter in module.parameters()):
        raise RuntimeError("Nonfinite PPO parameters")
    state["updates"] += 1
    row = {"update": state["updates"], "phase": "extension" if extension else "prefix",
           "decisions": len(buffer), "decision_reward_sum": reward_sum,
           "completed_episode_reward_sum": natural_reward_sum,
           "completed_episode_count": natural_count, "ended_episode_count": ended_count,
           "truncated_episode_count": truncated_count,
           "truncated_episode_reward_sum": truncated_reward_sum,
           "target_tail_zero_bootstrap": target_terminal, "bootstrap_value": next_value, **stats}
    state["updates_log"].append(row)
    return row


def full_state(agent, env, state: dict, metadata: dict) -> dict:
    return {"format": "matching_mappo_training_state_v1", "metadata": metadata,
            "actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(),
            "optimizer": agent.optimizer.state_dict(), "environment_state": vars(env),
            "agent_runtime": {k: v for k, v in vars(agent).items()
                              if k not in {"actor", "critic", "optimizer"}},
            "parameter_grads": {
                prefix: {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                         for name, parameter in module.named_parameters()}
                for prefix, module in (("actor", agent.actor), ("critic", agent.critic))},
            "training_state": state, "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_full_state(payload: dict, agent, env, metadata: dict) -> dict:
    if (payload.get("format") != "matching_mappo_training_state_v1" or
            payload.get("metadata") != metadata):
        raise ValueError("Resume protocol differs: configuration, source, seed, runtime or budget changed")
    agent.actor.load_state_dict(payload["actor"], strict=True)
    agent.critic.load_state_dict(payload["critic"], strict=True)
    agent.optimizer.load_state_dict(payload["optimizer"])
    for key, value in payload["agent_runtime"].items():
        setattr(agent, key, value)
    for prefix, module in (("actor", agent.actor), ("critic", agent.critic)):
        for name, parameter in module.named_parameters():
            gradient = payload["parameter_grads"][prefix][name]
            parameter.grad = None if gradient is None else gradient.to(agent.device).clone()
    env.__dict__.clear()
    env.__dict__.update(payload["environment_state"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["training_state"]


def training_summary(state: dict, metadata: dict, complete: bool) -> dict:
    return {"scenario": metadata["scenario"], "seed": metadata["train_seed"],
            "instance_seed": metadata["train_start_seed"], "updates": state["updates"],
            "rollout_steps": metadata["rollout_steps"],
            "completed_training_episodes": state["completed_training_episodes"],
            "ended_training_episodes": state["ended_training_episodes"],
            "truncated_training_episodes": state["truncated_training_episodes"],
            "total_training_decisions": state["total_training_decisions"],
            "algorithm": metadata["algorithm"], "environment": metadata["environment"],
            "paper_metadata": metadata, "max_replay_error": state["max_replay_error"],
            "invalid_actions": 0, "updates_log": state["updates_log"],
            "episodes": state["episodes"], "training_complete": complete,
            "status": "complete" if complete else "in_progress"}


def train(scenario: str, seed: int, instance_seed: int | None, updates: int | None,
          rollout_steps: int, output: Path, config: OursConfig, instance_path: str | None = None,
          *, prefix_updates: int = 300, target_episodes: int = 1200,
          smoke: bool = False, resume: bool = False,
          stop_after_updates: int | None = None) -> Path:
    """Save complete state at every PPO boundary; --resume never warm-starts."""
    if updates is not None and not smoke:
        raise ValueError("--updates is a fixed-update smoke budget; add --smoke")
    if smoke and updates is None:
        raise ValueError("--smoke requires an explicit --updates limit")
    if stop_after_updates is not None and (not smoke or stop_after_updates < 1):
        raise ValueError("--stop-after-updates requires --smoke and a positive limit")
    budget = TrainingBudget(prefix_updates, target_episodes, rollout_steps, updates)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        existing = [p for p in output.iterdir() if p.name != ".training.lock"]
        latest = output / "latest_state.pt"
        if existing and not resume:
            raise FileExistsError("Output contains an earlier run; use --resume or a new --output")
        if resume and not latest.is_file():
            raise FileNotFoundError("--resume requires this run's latest_state.pt")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        start, end = training_seed_range(scenario, seed, instance_seed)
        env = make_environment(scenario, start, instance_path)
        validate_environment(env)
        agent = make_agent(env, config)
        metadata = checkpoint_metadata(scenario, seed, start, end, updates,
                                       rollout_steps, asdict(env.config), config, instance_path)
        formal = (not smoke and (prefix_updates, target_episodes, rollout_steps) == (300, 1200, 512)
                  and seed in TRAIN_SEEDS and instance_seed is None and config == OursConfig()
                  and reference_instance_matches(scenario, instance_path))
        metadata["budget"] = {
            "kind": "smoke_fixed_updates" if updates is not None else "prefix_then_minimum_natural_episodes",
            "classification": "paper" if formal else "smoke" if smoke else "custom",
            "prefix_updates": prefix_updates, "target_episodes": target_episodes,
            "smoke_updates": updates, "rollout_steps": rollout_steps,
            "prefix_preserves_actual_episodes_above_target": True,
            "extension_stops_at_first_natural_target": True,
            "extension_tail_zero_bootstrap": True,
            "truncation_counts_toward_target": False, "unfinished_penalty": 0.0,
            "seed_partition_cycling": "extension_only_by_ended_episode_cursor",
        }
        # Metadata records the budget; actual counts live in training_stage/training.json.
        metadata["joint_training_decisions"] = None if updates is None else updates * rollout_steps
        metadata["updates"] = updates
        state = (restore_full_state(torch.load(latest, map_location="cpu", weights_only=False),
                                    agent, env, metadata) if resume else initial_state(env, start))
        atomic_json(output / "protocol.json", metadata)
        if not resume:
            # A first-rollout interruption can recover the exact initialized run.
            atomic_save(latest, full_state(agent, env, state, metadata))
            atomic_json(output / "training.json", training_summary(state, metadata, False))
        while budget.needs_update(state):
            row = collect_update(agent, env, state, budget, start, end)
            complete = not budget.needs_update(state)
            atomic_save(latest, full_state(agent, env, state, metadata))
            atomic_json(output / "training.json", training_summary(state, metadata, complete))
            average = (row["completed_episode_reward_sum"] / row["completed_episode_count"]
                       if row["completed_episode_count"] else None)
            print(json.dumps({"update": state["updates"], "phase": row["phase"],
                              "episodes": state["completed_training_episodes"],
                              "target_episodes": target_episodes, "mean_episode_reward": average}), flush=True)
            if stop_after_updates is not None and state["updates"] >= stop_after_updates and not complete:
                return latest
        checkpoint = output / "checkpoint.pt"
        inference = {"format": "ours_joint_matching_ppo_v1",
                     "initialization_profile": agent.config.initialization_profile,
                     "actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(),
                     "config": asdict(config), "obs_dim": agent.obs_dim, "state_dim": agent.state_dim,
                     "action_dim": agent.action_dim, "num_agents": agent.num_agents,
                     "max_jobs": agent.max_jobs, "paper_metadata": metadata,
                     "training_stage": {"completed_updates": state["updates"], "final_checkpoint": True,
                                        "completed_training_episodes": state["completed_training_episodes"],
                                        "total_training_decisions": state["total_training_decisions"]}}
        atomic_save(checkpoint, inference)
        atomic_json(output / "training.json", training_summary(state, metadata, True))
        return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional YAML configuration")
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--instance-seed", type=int, help="Custom training-pool override")
    parser.add_argument("--prefix-updates", type=int)
    parser.add_argument("--target-episodes", type=int)
    parser.add_argument("--updates", type=int, help="Fixed updates for an explicit --smoke run")
    parser.add_argument("--smoke", action="store_true", help="Mark a short verification run; requires --updates")
    parser.add_argument("--stop-after-updates", type=int, help="Smoke interruption after a saved PPO boundary")
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--resume", action="store_true", help="Restore this output's complete local training state")
    parser.add_argument("--instance", help="External .fjs or JSON file for Mk08/Mk09")
    parser.add_argument("--output", type=Path, required=True, help="Directory for all generated outputs")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text()) if args.config else {}
    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a mapping")
    if "updates" in settings and not args.smoke:
        parser.error("The legacy updates key requires --smoke; use prefix_updates/target_episodes")
    scenario = args.scenario or settings.get("scenario", "S20L")
    seed = args.seed if args.seed is not None else int(settings.get("seed", 2400001))
    instance_seed = args.instance_seed if args.instance_seed is not None else settings.get("instance_seed")
    config = OursConfig(**settings.get("algorithm", {}))
    chosen_updates = args.updates if args.updates is not None else settings.get("updates")
    result = train(
        scenario, int(seed), int(instance_seed) if instance_seed is not None else None,
        int(chosen_updates) if chosen_updates is not None else None,
        args.rollout_steps if args.rollout_steps is not None else int(settings.get("rollout_steps", 512)),
        args.output, config, args.instance or settings.get("instance"),
        prefix_updates=args.prefix_updates if args.prefix_updates is not None else int(settings.get("prefix_updates", 300)),
        target_episodes=args.target_episodes if args.target_episodes is not None else int(settings.get("target_episodes", 1200)),
        smoke=args.smoke, resume=args.resume, stop_after_updates=args.stop_after_updates,
    )
    print(f"{'Checkpoint' if result.name == 'checkpoint.pt' else 'Training state'} saved to {result}")


if __name__ == "__main__":
    main()
