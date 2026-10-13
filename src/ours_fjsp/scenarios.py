"""Portable problem configurations for the five Ours scenarios."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .envs import FJSPEnvConfig, make_fjsp_env


@dataclass(frozen=True)
class Scenario:
    jobs: int
    machines: int
    breakdown_rate: float
    max_steps: int
    first_instance_seed: int
    requires_instance: bool = False


SCENARIOS = {
    "S20L": Scenario(20, 10, .01, 2000, 2410000),
    "S20H": Scenario(20, 10, .02, 2000, 2420000),
    "S40H": Scenario(40, 10, .02, 4000, 2430000),
    "Mk08": Scenario(20, 10, .02, 5000, 2440000, True),
    "Mk09": Scenario(20, 10, .02, 5000, 2450000, True),
}


def environment_config(scenario: str, seed: int, instance_path: str | None = None) -> FJSPEnvConfig:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {', '.join(SCENARIOS)}")
    spec = SCENARIOS[scenario]
    if spec.requires_instance and not instance_path:
        raise ValueError(f"{scenario} requires --instance pointing to a user-provided .fjs or JSON problem")
    if instance_path and not Path(instance_path).is_file():
        raise FileNotFoundError(instance_path)
    if not spec.requires_instance and instance_path:
        raise ValueError("--instance is intended for Mk08/Mk09; the S scenarios generate instances")
    return FJSPEnvConfig(
        env_type="dynamic_benchmark" if spec.requires_instance else "dynamic",
        seed=int(seed), instance_path=instance_path,
        num_jobs=spec.jobs, num_machines=spec.machines,
        ops_per_job_min=1 if spec.requires_instance else 5,
        ops_per_job_max=14 if spec.requires_instance else 8,
        eligible_machines_min=2, eligible_machines_max=5,
        processing_time_min=1, processing_time_max=30,
        op_feature_dim=15, feature_profile="tardiness_only",
        urgency_mode="remaining_min", hide_unarrived_jobs=True,
        separate_active_repair_feature=True,
        dynamic_arrival_mode="poisson", arrival_rate=.35,
        due_date_factor=1.7, due_date_noise=.25,
        machine_breakdown=True, breakdown_rate=spec.breakdown_rate,
        failure_horizon_multiplier=10.0,
        repair_time_min=3.0, repair_time_max=12.0,
        energy_rate_min=.7, energy_rate_max=1.8,
        max_steps=spec.max_steps,
        reward_weights={
            "time": 0.0, "makespan": 0.0, "tardiness": 0.0,
            "energy": 0.0, "load_balance": 0.0, "load_balance_dense": 0.0,
            "invalid": 0.0, "wait": 0.0, "assigned": 0.0,
            "operation_completed": 0.0, "job_completed": 0.0,
            "delta_tardiness": 1.0, "processing_time": 0.0,
            "unfinished_ops_time": 0.0,
        },
    )


def make_environment(scenario: str = "S20L", seed: int = 2410000, instance_path: str | None = None):
    return make_fjsp_env(environment_config(scenario, seed, instance_path))
