"""Record and validate the paper method used to produce a checkpoint."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import platform
from pathlib import Path

import numpy as np
import torch

from .algorithms import OursConfig
from .algorithms.initialization import INITIALIZATION_PROFILE
from .scenarios import SCENARIOS


PAPER_CONTRACT = {
    "version": 1,
    "urgency_mode": "paper_formula_sum",
    "wait_pool": "operations_with_any_currently_feasible_machine",
    "cumulative_load": True,
    "load_machine_feature_index": 3,
    "graph_messages": True,
    "wait_scores": True,
    "action_policy": "exact",
    "allow_partial": True,
    "nonempty_when_feasible": True,
    "reward": "negative_incremental_total_tardiness",
    "round_reset": False,
    "initial_checkpoint": None,
    "initialization_profile": INITIALIZATION_PROFILE,
}
TRAIN_SEEDS = (2400001, 2400002, 2400003)


def training_seed_range(scenario: str, seed: int, override: int | None = None):
    """Use disjoint thirds of each scenario's 10,000-instance training pool."""
    base = SCENARIOS[scenario].first_instance_seed
    if override is not None:
        return int(override), None
    if seed not in TRAIN_SEEDS:
        return base, base + 9999
    index = TRAIN_SEEDS.index(seed)
    return base + 10000 * index // 3, base + 10000 * (index + 1) // 3 - 1


def validate_environment(env) -> None:
    config = env.config
    active_rewards = {k: v for k, v in config.reward_weights.items() if v != 0}
    if (config.urgency_mode != "paper_formula_sum" or
            not config.paper_include_cumulative_load or
            config.feature_profile != "tardiness_only" or
            not config.hide_unarrived_jobs or
            not config.separate_active_repair_feature or
            config.op_feature_dim != 15 or
            active_rewards != {"delta_tardiness": 1.0}):
        raise ValueError("Environment does not implement the paper method")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_metadata(scenario: str, seed: int, start: int, end: int | None,
                        updates: int | None, rollout_steps: int, environment: dict,
                        config: OursConfig, instance_path: str | None) -> dict:
    package = Path(__file__).parent
    return {
        "paper_contract": dict(PAPER_CONTRACT),
        "scenario": scenario,
        "train_seed": int(seed),
        "train_start_seed": start,
        "train_end_seed": end,
        "updates": updates,
        "rollout_steps": rollout_steps,
        "joint_training_decisions": None if updates is None else updates * rollout_steps,
        "environment": environment,
        "algorithm": asdict(config),
        "instance_sha256": file_sha256(Path(instance_path)) if instance_path else None,
        "source_sha256": {str(p.relative_to(package)): file_sha256(p)
                          for p in sorted(package.rglob("*.py"))},
        "runtime": {"python": platform.python_version(), "numpy": str(np.__version__),
                    "torch": str(torch.__version__)},
    }


def checkpoint_config(saved: dict) -> OursConfig:
    values = saved.get("config")
    if not isinstance(values, dict):
        raise ValueError("Checkpoint must contain the Ours algorithm configuration")
    values = dict(values)
    # Evaluation uses CPU; the saved training configuration remains in metadata.
    values["device"] = "cpu"
    return OursConfig(**values)


def validate_checkpoint_protocol(saved: dict, scenario: str, env) -> None:
    """Reject ambiguous or static-urgency checkpoints, even with matching shapes."""
    validate_environment(env)
    metadata = saved.get("paper_metadata", {})
    if metadata.get("paper_contract") != PAPER_CONTRACT:
        raise ValueError("Checkpoint lacks the current paper-method metadata")
    if metadata.get("scenario") != scenario:
        raise ValueError("Checkpoint scenario differs from the requested scenario")
    old = dict(metadata.get("environment", {}))
    current = asdict(env.config)
    # Evaluation changes the instance seed and may relocate the same benchmark.
    for field in ("seed", "instance_path"):
        old.pop(field, None)
        current.pop(field, None)
    if old != current:
        raise ValueError("Checkpoint environment differs from the evaluation environment")
    path = env.config.instance_path
    actual_hash = file_sha256(Path(path)) if path else None
    if metadata.get("instance_sha256") != actual_hash:
        raise ValueError("Benchmark instance differs from the training instance")
