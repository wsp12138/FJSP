from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .instances import (
    FJSPInstance,
    generate_random_instance,
    inject_dynamic_attributes,
    load_builtin_instance,
    load_instance,
)
from .benchmark_parsers import load_benchmark_instance


@dataclass
class FJSPEnvConfig:
    env_type: str = "static"
    seed: int = 0
    instance_path: str | None = None
    builtin_instance: str = "toy_static"
    num_jobs: int = 10
    num_machines: int = 5
    ops_per_job_min: int = 3
    ops_per_job_max: int = 5
    eligible_machines_min: int = 1
    eligible_machines_max: int = 3
    processing_time_min: int = 1
    processing_time_max: int = 20
    arrival_mode: str = "zero"
    arrival_rate: float = 0.25
    due_date_factor: float = 1.6
    due_date_noise: float = 0.15
    max_steps: int = 1000
    idle_wait_time: float = 1.0
    op_feature_dim: int = 10
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "time": 1.0,
            "makespan": 0.1,
            "tardiness": 1.0,
            "energy": 0.02,
            "load_balance": 0.05,
            "load_balance_dense": 0.0,
            "invalid": 0.2,
            "wait": 0.02,
            "assigned": 0.0,
            "operation_completed": 0.0,
            "job_completed": 0.0,
            "delta_tardiness": 0.0,
            "processing_time": 0.0,
            "unfinished_ops_time": 0.0,
        }
    )
    dynamic_arrival_mode: str = "poisson"
    machine_breakdown: bool = False
    breakdown_rate: float = 0.02
    repair_time_min: float = 2.0
    repair_time_max: float = 8.0
    energy_rate_min: float = 0.8
    energy_rate_max: float = 1.5
    # Appended for positional compatibility with historical FJSPEnvConfig calls.
    # ``remaining_min`` is the controlled ablation: it keeps the static route
    # estimate but uses the same observable machine-delay feature as the full
    # method. ``legacy_remaining_min`` exactly preserves historical observations,
    # including their precomputed busy-until value, for old checkpoints only.
    urgency_mode: str = "legacy_remaining_min"
    # New single-objective experiments hide channels retained only for the
    # deleted energy/load/weighted-tardiness objectives while preserving tensor
    # shapes for fair ablations. ``legacy`` keeps old checkpoints reproducible.
    feature_profile: str = "legacy"
    # Historical experiments generated breakdowns over 2.5 ideal-load
    # horizons. New protocols can extend that window without changing old runs.
    failure_horizon_multiplier: float = 2.5
    # Causal online policies must not observe routes, processing times, due
    # dates, eligibility, or normalization constants from jobs that have not
    # arrived.  Disabled by default so historical checkpoints remain exactly
    # reproducible.
    hide_unarrived_jobs: bool = False
    # ECOT needs to ablate only an *active repair* while retaining observable
    # unfinished processing.  Historical observations combine those quantities
    # in one machine-ready-delay channel, so this opt-in profile adds one
    # machine feature and exposes them separately.  The default preserves every
    # legacy observation/state shape and value.
    separate_active_repair_feature: bool = False


class BaseFJSPEnv:
    """Event-based flexible job-shop scheduling environment.

    Agents are machines. At each decision epoch, every idle and available
    machine chooses one ready operation or waits. The environment preserves
    operation precedence, machine eligibility, release times, and optional
    machine breakdown windows.
    """

    def __init__(self, config: FJSPEnvConfig):
        self.config = config
        if self.config.urgency_mode not in {
            "legacy_remaining_min",
            "remaining_min",
            "availability_aware",
            "separated_debt",
            "repair_contrast",
        }:
            raise ValueError(
                "urgency_mode must be 'legacy_remaining_min', 'remaining_min', "
                "'availability_aware', 'separated_debt', or 'repair_contrast'"
            )
        if self.config.feature_profile not in {"legacy", "tardiness_only"}:
            raise ValueError(
                "feature_profile must be 'legacy' or 'tardiness_only'"
            )
        if self.config.urgency_mode in {"separated_debt", "repair_contrast"}:
            if self.config.feature_profile != "tardiness_only":
                raise ValueError(
                    f"urgency_mode={self.config.urgency_mode!r} requires "
                    "feature_profile='tardiness_only'"
                )
            if int(self.config.op_feature_dim) < 15:
                raise ValueError(
                    f"urgency_mode={self.config.urgency_mode!r} stores its "
                    "repair signal in "
                    "operation feature 14 and therefore requires op_feature_dim>=15"
                )
        if self.config.failure_horizon_multiplier <= 0.0:
            raise ValueError("failure_horizon_multiplier must be positive")
        self.rng = np.random.default_rng(config.seed)
        self.instance: FJSPInstance = self._make_instance()
        self.num_agents = self.instance.num_machines
        self.max_jobs = max(config.num_jobs, self.instance.num_jobs)
        self.max_ops_per_job = max(config.ops_per_job_max, self.instance.max_ops_per_job)
        self.max_operations = max(self.max_jobs * self.max_ops_per_job, self.instance.num_operations)
        self.action_dim = self.max_operations + 1
        self.op_feature_dim = max(10, int(config.op_feature_dim))
        self.machine_feature_dim = (
            5 if self.config.separate_active_repair_feature else 4
        )
        self.global_feature_dim = 6
        self.obs_dim = (
            self.global_feature_dim
            + self.num_agents
            + self.num_agents * self.machine_feature_dim
            + self.max_operations * self.op_feature_dim
        )
        self.state_dim = (
            self.global_feature_dim
            + self.num_agents * self.machine_feature_dim
            + self.max_operations * (self.op_feature_dim + self.num_agents)
        )
        self.reset()

    def _make_instance(self) -> FJSPInstance:
        raise NotImplementedError

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.instance = self._make_instance()
        self.num_agents = self.instance.num_machines
        self._build_operation_index()

        self.time = 0.0
        self.step_count = 0
        self.done = False
        self.op_status = np.zeros(self.num_ops, dtype=np.int8)
        self.op_start = np.full(self.num_ops, np.nan, dtype=np.float32)
        self.op_completion = np.full(self.num_ops, np.nan, dtype=np.float32)
        self.op_machine = np.full(self.num_ops, -1, dtype=np.int32)
        self.machine_busy_until = np.zeros(self.num_agents, dtype=np.float32)
        self.machine_current_op = np.full(self.num_agents, -1, dtype=np.int32)
        self.machine_workload = np.zeros(self.num_agents, dtype=np.float32)
        self.energy_used = 0.0
        self.invalid_actions = 0
        self.machine_energy_rates = self._make_energy_rates()
        self.failure_windows = self._make_failure_windows()
        self._advance_to_next_decision()
        return self._obs_state_mask()

    def _build_operation_index(self) -> None:
        self.op_keys = self.instance.operation_keys()
        self.num_ops = len(self.op_keys)
        self.key_to_op = {key: idx for idx, key in enumerate(self.op_keys)}
        self.job_last_op = {
            job_id: self.key_to_op[(job_id, len(job.operations) - 1)]
            for job_id, job in enumerate(self.instance.jobs)
        }
        self.job_op_counts = [len(job.operations) for job in self.instance.jobs]

    def _make_energy_rates(self) -> np.ndarray:
        c = self.config
        return self.rng.uniform(c.energy_rate_min, c.energy_rate_max, self.num_agents)

    def _make_failure_windows(self) -> list[list[tuple[float, float]]]:
        if not self.config.machine_breakdown:
            self.failure_generation_horizon = 0.0
            return [[] for _ in range(self.num_agents)]
        windows: list[list[tuple[float, float]]] = []
        rough_horizon = sum(
            self.instance.min_total_processing_time(j)
            for j in range(self.instance.num_jobs)
        ) / max(1, self.num_agents)
        rough_horizon *= float(self.config.failure_horizon_multiplier)
        self.failure_generation_horizon = float(rough_horizon)
        for _ in range(self.num_agents):
            t = float(self.rng.exponential(1.0 / max(self.config.breakdown_rate, 1e-6)))
            machine_windows: list[tuple[float, float]] = []
            while t < rough_horizon:
                repair = float(
                    self.rng.uniform(
                        self.config.repair_time_min, self.config.repair_time_max
                    )
                )
                machine_windows.append((t, t + repair))
                t += repair + float(
                    self.rng.exponential(1.0 / max(self.config.breakdown_rate, 1e-6))
                )
            windows.append(machine_windows)
        return windows

    def _operation(self, op_idx: int):
        job_id, op_id = self.op_keys[op_idx]
        return self.instance.jobs[job_id].operations[op_id]

    def _job_arrived(self, job_id: int) -> bool:
        return self.instance.jobs[job_id].arrival_time <= self.time + 1e-9

    def _observation_job_ids(self) -> list[int]:
        """Jobs permitted to contribute to the current observation."""

        if not self.config.hide_unarrived_jobs:
            return list(range(self.instance.num_jobs))
        return [
            job_id
            for job_id in range(self.instance.num_jobs)
            if self._job_arrived(job_id)
        ]

    def _observation_total_min_processing(self) -> float:
        return float(
            sum(
                self.instance.min_total_processing_time(job_id)
                for job_id in self._observation_job_ids()
            )
        )

    def _observation_scale(self) -> float:
        return max(
            1.0,
            self._observation_total_min_processing() / max(1, self.num_agents),
        )

    def _observation_operation_count(self) -> int:
        if not self.config.hide_unarrived_jobs:
            return int(self.num_ops)
        return int(
            sum(
                self.job_op_counts[job_id]
                for job_id in self._observation_job_ids()
            )
        )

    def _predecessor_complete(self, job_id: int, op_id: int) -> bool:
        if op_id == 0:
            return True
        pred_idx = self.key_to_op[(job_id, op_id - 1)]
        return self.op_status[pred_idx] == 2

    def _ready_operations(self) -> list[int]:
        ready: list[int] = []
        for op_idx, (job_id, op_id) in enumerate(self.op_keys):
            if self.op_status[op_idx] != 0:
                continue
            if self._job_arrived(job_id) and self._predecessor_complete(job_id, op_id):
                ready.append(op_idx)
        return ready

    def _in_failure_window(self, machine: int, time_value: float | None = None) -> bool:
        t = self.time if time_value is None else time_value
        return any(start <= t < end for start, end in self.failure_windows[machine])

    def _failure_window_end(self, machine: int, time_value: float | None = None) -> float | None:
        t = self.time if time_value is None else time_value
        for start, end in self.failure_windows[machine]:
            if start <= t < end:
                return end
        return None

    def _machine_available(self, machine: int) -> bool:
        return (
            self.machine_busy_until[machine] <= self.time + 1e-9
            and not self._in_failure_window(machine)
        )

    def _decision_available(self) -> bool:
        ready = self._ready_operations()
        if not ready:
            return False
        for machine in range(self.num_agents):
            if not self._machine_available(machine):
                continue
            for op_idx in ready:
                if machine in self._operation(op_idx).machine_times:
                    return True
        return False

    def _update_completed_operations(self) -> None:
        for machine in range(self.num_agents):
            op_idx = self.machine_current_op[machine]
            if op_idx >= 0 and self.machine_busy_until[machine] <= self.time + 1e-9:
                self.op_status[op_idx] = 2
                self.op_completion[op_idx] = self.machine_busy_until[machine]
                self.machine_current_op[machine] = -1

    def _next_job_arrival(self) -> float | None:
        future = [
            job.arrival_time
            for job in self.instance.jobs
            if job.arrival_time > self.time + 1e-9
        ]
        return min(future) if future else None

    def _next_failure_boundary(self) -> float | None:
        future: list[float] = []
        for machine in range(self.num_agents):
            end = self._failure_window_end(machine)
            if end is not None and end > self.time + 1e-9:
                future.append(end)
            for start, _ in self.failure_windows[machine]:
                if start > self.time + 1e-9:
                    future.append(start)
                    break
        return min(future) if future else None

    def _next_completion(self) -> float | None:
        busy = [
            float(t)
            for t, op in zip(self.machine_busy_until, self.machine_current_op)
            if op >= 0 and t > self.time + 1e-9
        ]
        return min(busy) if busy else None

    def _advance_to_next_decision(self) -> None:
        while not self._all_done() and not self._decision_available():
            candidates = [
                t
                for t in [
                    self._next_completion(),
                    self._next_job_arrival(),
                    self._next_failure_boundary(),
                ]
                if t is not None and t > self.time + 1e-9
            ]
            if not candidates:
                break
            self.time = min(candidates)
            self._update_completed_operations()
        self._update_completed_operations()

    def _adjust_end_for_failures(self, machine: int, start: float, duration: float) -> float:
        actual_start = start
        for w_start, w_end in self.failure_windows[machine]:
            if w_start <= actual_start < w_end:
                actual_start = w_end
        end = actual_start + duration
        for w_start, w_end in self.failure_windows[machine]:
            if w_end <= actual_start:
                continue
            if w_start < end and w_end > actual_start:
                end += w_end - max(w_start, actual_start)
        return end

    def _assign(self, machine: int, op_idx: int) -> float:
        op = self._operation(op_idx)
        proc_time = float(op.machine_times[machine])
        start = self.time
        end = self._adjust_end_for_failures(machine, start, proc_time)
        self.op_status[op_idx] = 1
        self.op_start[op_idx] = start
        self.op_machine[op_idx] = machine
        self.machine_current_op[machine] = op_idx
        self.machine_busy_until[machine] = end
        self.machine_workload[machine] += proc_time
        self.energy_used += proc_time * self.machine_energy_rates[machine]
        return proc_time

    def step(
        self, actions: np.ndarray | list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() called after episode finished. Call reset().")
        old_time = self.time
        old_energy = self.energy_used
        old_completed_ops = int(np.sum(self.op_status == 2))
        old_unfinished_ops = self.num_ops - old_completed_ops
        old_completed_jobs = self._completed_jobs_count()
        old_tardiness = self._total_tardiness()
        old_load_std = float(np.std(self.machine_workload))
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if actions.shape[0] != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {actions.shape[0]}")

        masks = self.action_masks()
        requested: dict[int, list[int]] = {}
        invalid = 0
        waits = 0
        for machine, action in enumerate(actions):
            if action <= 0:
                waits += int(self._machine_available(machine))
                continue
            if action >= self.action_dim or not masks[machine, action]:
                invalid += 1
                continue
            op_idx = int(action - 1)
            requested.setdefault(op_idx, []).append(machine)

        assigned = 0
        assigned_proc_time = 0.0
        for op_idx, machines in requested.items():
            op = self._operation(op_idx)
            chosen = min(
                machines,
                key=lambda m: (op.machine_times.get(m, float("inf")), self.machine_workload[m], m),
            )
            assigned_proc_time += self._assign(chosen, op_idx)
            assigned += 1

        if assigned == 0:
            next_events = [
                t
                for t in [
                    self._next_completion(),
                    self._next_job_arrival(),
                    self._next_failure_boundary(),
                ]
                if t is not None and t > self.time + 1e-9
            ]
            self.time = min(next_events) if next_events else self.time + self.config.idle_wait_time
            self._update_completed_operations()
        else:
            self._advance_to_next_decision()

        self.step_count += 1
        self.invalid_actions += invalid
        self.done = self._all_done() or self.step_count >= self.config.max_steps
        if self.done and not self._all_done():
            # Truncated episodes receive a penalty but still expose current metrics.
            self.done = True
        new_completed_ops = int(np.sum(self.op_status == 2))
        new_completed_jobs = self._completed_jobs_count()
        new_tardiness = self._total_tardiness()
        operation_completed = max(0, new_completed_ops - old_completed_ops)
        job_completed = max(0, new_completed_jobs - old_completed_jobs)
        delta_tardiness = max(0.0, new_tardiness - old_tardiness)

        reward_scalar = self._reward(
            old_time=old_time,
            old_energy=old_energy,
            assigned=assigned,
            operation_completed=operation_completed,
            job_completed=job_completed,
            delta_tardiness=delta_tardiness,
            assigned_proc_time=assigned_proc_time,
            old_unfinished_ops=old_unfinished_ops,
            old_load_std=old_load_std,
            invalid=invalid,
            waits=waits,
            terminal=self.done,
        )
        reward_components = self._reward_components(
            old_time=old_time,
            old_energy=old_energy,
            assigned=assigned,
            operation_completed=operation_completed,
            job_completed=job_completed,
            delta_tardiness=delta_tardiness,
            assigned_proc_time=assigned_proc_time,
            old_unfinished_ops=old_unfinished_ops,
            old_load_std=old_load_std,
            invalid=invalid,
            waits=waits,
            terminal=self.done,
        )
        rewards = np.full(self.num_agents, reward_scalar, dtype=np.float32)
        obs, state, next_masks = self._obs_state_mask()
        info = self.metrics()
        info.update({
            "assigned": assigned,
            "operation_completed": operation_completed,
            "job_completed": job_completed,
            "delta_tardiness": delta_tardiness,
            "assigned_proc_time": assigned_proc_time,
            "load_std_delta": float(np.std(self.machine_workload)) - old_load_std,
            "invalid": invalid,
            "waits": waits,
            "reward_components": reward_components,
        })
        return obs, state, rewards, next_masks, self.done, info

    def _reward(
        self,
        *,
        old_time: float,
        old_energy: float,
        assigned: int,
        operation_completed: int,
        job_completed: int,
        delta_tardiness: float,
        assigned_proc_time: float,
        old_unfinished_ops: int,
        old_load_std: float,
        invalid: int,
        waits: int,
        terminal: bool,
    ) -> float:
        return float(
            sum(
                self._reward_components(
                    old_time=old_time,
                    old_energy=old_energy,
                    assigned=assigned,
                    operation_completed=operation_completed,
                    job_completed=job_completed,
                    delta_tardiness=delta_tardiness,
                    assigned_proc_time=assigned_proc_time,
                    old_unfinished_ops=old_unfinished_ops,
                    old_load_std=old_load_std,
                    invalid=invalid,
                    waits=waits,
                    terminal=terminal,
                ).values()
            )
        )

    def _reward_components(
        self,
        *,
        old_time: float,
        old_energy: float,
        assigned: int,
        operation_completed: int,
        job_completed: int,
        delta_tardiness: float,
        assigned_proc_time: float,
        old_unfinished_ops: int,
        old_load_std: float,
        invalid: int,
        waits: int,
        terminal: bool,
    ) -> dict[str, float]:
        w = self.config.reward_weights
        components = {
            "assigned": w.get("assigned", 0.0) * assigned,
            "operation_completed": w.get("operation_completed", 0.0) * operation_completed,
            "job_completed": w.get("job_completed", 0.0) * job_completed,
            "delta_tardiness": -w.get("delta_tardiness", 0.0) * delta_tardiness,
            "processing_time": -w.get("processing_time", 0.0) * assigned_proc_time,
            "unfinished_ops_time": -w.get("unfinished_ops_time", 0.0)
            * max(0.0, self.time - old_time)
            * (old_unfinished_ops / max(1, self.num_ops)),
            "time": -w.get("time", 1.0) * max(0.0, self.time - old_time),
            "energy": -w.get("energy", 0.0) * max(0.0, self.energy_used - old_energy),
            "invalid": -w.get("invalid", 0.0) * invalid,
            "wait": -w.get("wait", 0.0) * waits,
            "makespan": 0.0,
            "tardiness": 0.0,
            # Potential-based dense shaping exposes the final LoadStd objective
            # at every assignment. With gamma near one, these differences
            # telescope to a penalty on the final load dispersion.
            "load_balance": w.get("load_balance_dense", 0.0)
            * (old_load_std - float(np.std(self.machine_workload))),
        }
        if terminal:
            metrics = self.metrics()
            components["makespan"] = -w.get("makespan", 0.0) * metrics["makespan"]
            components["tardiness"] = -w.get("tardiness", 0.0) * metrics["total_tardiness"]
            components["load_balance"] += -w.get("load_balance", 0.0) * metrics["load_std"]
        return {key: float(value) for key, value in components.items()}

    def _all_done(self) -> bool:
        return bool(np.all(self.op_status == 2)) if self.num_ops else True

    def _job_completion_time(self, job_id: int) -> float:
        last_op = self.job_last_op[job_id]
        if np.isnan(self.op_completion[last_op]):
            return self.time
        return float(self.op_completion[last_op])

    def _completed_jobs_count(self) -> int:
        return int(
            sum(self.op_status[self.job_last_op[job_id]] == 2 for job_id in range(self.instance.num_jobs))
        )

    def _total_tardiness(self) -> float:
        total_tardiness = 0.0
        for job_id, job in enumerate(self.instance.jobs):
            if job.due_date is None:
                continue
            completion = self._job_completion_time(job_id)
            total_tardiness += max(0.0, completion - job.due_date)
        return float(total_tardiness)

    def metrics(self) -> dict[str, Any]:
        completed = int(np.sum(self.op_status == 2))
        makespan = float(np.nanmax(self.op_completion)) if completed else float(self.time)
        total_tardiness = self._total_tardiness()
        weighted_tardiness = 0.0
        for job_id, job in enumerate(self.instance.jobs):
            if job.due_date is None:
                continue
            completion = self._job_completion_time(job_id)
            weighted_tardiness += job.weight * max(0.0, completion - job.due_date)
        return {
            "time": float(self.time),
            "makespan": makespan,
            "completed_operations": completed,
            "total_operations": self.num_ops,
            "total_tardiness": float(total_tardiness),
            "weighted_tardiness": float(weighted_tardiness),
            "energy": float(self.energy_used),
            "load_std": float(np.std(self.machine_workload)),
            "machine_workload": self.machine_workload.tolist(),
            "done": self.done,
            "steps": self.step_count,
        }

    def bottleneck_targets(
        self, max_jobs: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Auxiliary critic targets for bottleneck-aware value learning.

        Machine target: normalized accumulated workload plus current busy
        remainder. This behaves like a bottleneck price signal.

        Job target: normalized urgency pressure, based on the configured
        remaining-work estimate relative to slack before the due date.
        """

        machine_pressure = np.zeros(self.num_agents, dtype=np.float32)
        for machine in range(self.num_agents):
            if self.config.urgency_mode != "legacy_remaining_min":
                busy_remaining = self._machine_ready_delay(machine)
            else:
                busy_remaining = max(
                    0.0, float(self.machine_busy_until[machine] - self.time)
                )
            machine_pressure[machine] = self.machine_workload[machine] + busy_remaining
        denom = float(np.max(machine_pressure))
        if denom > 1e-9:
            machine_pressure = machine_pressure / denom

        target_jobs = max_jobs or self.max_jobs
        job_pressure = np.zeros(target_jobs, dtype=np.float32)
        total_min_proc = sum(
            self.instance.min_total_processing_time(j)
            for j in range(self.instance.num_jobs)
        )
        scale = max(1.0, total_min_proc / max(1, self.num_agents))
        for job_id, job in enumerate(self.instance.jobs[:target_jobs]):
            if job_id >= target_jobs:
                break
            if self.op_status[self.job_last_op[job_id]] == 2:
                continue
            remaining = self._urgency_remaining_work(job_id)
            due = job.due_date if job.due_date is not None else self.time + scale
            slack = due - self.time - remaining
            job_pressure[job_id] = max(0.0, -slack) / scale
        max_pressure = float(np.max(job_pressure))
        if max_pressure > 1e-9:
            job_pressure = job_pressure / max_pressure
        return machine_pressure.astype(np.float32), job_pressure.astype(np.float32)

    def action_masks(self) -> np.ndarray:
        masks = np.zeros((self.num_agents, self.action_dim), dtype=bool)
        masks[:, 0] = True
        ready = self._ready_operations()
        for machine in range(self.num_agents):
            if not self._machine_available(machine):
                continue
            for op_idx in ready:
                if machine in self._operation(op_idx).machine_times:
                    masks[machine, op_idx + 1] = True
        return masks

    def _global_features(self) -> np.ndarray:
        completed = float(np.sum(self.op_status == 2))
        ready_count = float(len(self._ready_operations()))
        available_machines = float(sum(self._machine_available(m) for m in range(self.num_agents)))
        total_min_proc = self._observation_total_min_processing()
        scale = self._observation_scale()
        visible_operations = self._observation_operation_count()
        energy_feature = self.energy_used / max(1.0, total_min_proc)
        if self.config.feature_profile == "tardiness_only":
            energy_feature = 0.0
        return np.array(
            [
                self.time / scale,
                completed / max(1, visible_operations),
                ready_count / max(1, visible_operations),
                available_machines / max(1, self.num_agents),
                energy_feature,
                self.step_count / max(1, self.config.max_steps),
            ],
            dtype=np.float32,
        )

    def _remaining_min_work(self, job_id: int, start_op_id: int = 0) -> float:
        remaining = 0.0
        for op_id in range(start_op_id, self.job_op_counts[job_id]):
            op_idx = self.key_to_op[(job_id, op_id)]
            if self.op_status[op_idx] != 2:
                remaining += self._operation(op_idx).min_time()
        return float(remaining)

    def _machine_ready_delay(self, machine: int) -> float:
        """Observable delay until a machine can accept a new operation.

        Only the current busy interval and an active repair window are used. A
        future failure window is deliberately ignored so the urgency estimate
        does not expose future disturbances to the policy.
        """

        delay = self._machine_remaining_processing(machine)
        repair_end = self._failure_window_end(machine)
        if repair_end is not None:
            delay += max(0.0, float(repair_end - self.time))
        return float(delay)

    def _machine_active_repair_remaining(self, machine: int) -> float:
        """Return only the repair interval active at the current event time.

        ``_failure_window_end`` deliberately returns ``None`` for every future
        window, so this feature cannot reveal a scheduled future breakdown.
        """

        repair_end = self._failure_window_end(machine)
        if repair_end is None:
            return 0.0
        return max(0.0, float(repair_end - self.time))

    def _machine_remaining_processing(self, machine: int) -> float:
        """Observable unfinished processing, excluding an active repair delay."""

        remaining = 0.0
        op_idx = int(self.machine_current_op[machine])
        if op_idx >= 0:
            started = float(self.op_start[op_idx])
            processing_time = float(
                self._operation(op_idx).machine_times[machine]
            )
            elapsed = max(0.0, float(self.time - started))
            observed_downtime = 0.0
            for failure_start, failure_end in self.failure_windows[machine]:
                if failure_start >= self.time:
                    continue
                observed_downtime += max(
                    0.0,
                    min(float(failure_end), float(self.time))
                    - max(float(failure_start), started),
                )
            processed = max(0.0, elapsed - observed_downtime)
            remaining = max(0.0, processing_time - processed)
        return float(remaining)

    def _availability_aware_remaining_work(self, job_id: int) -> float:
        """Earliest sequential completion lower bound from current availability.

        The estimate propagates the current ready time of every eligible machine
        through the unfinished route of one job. It captures an active failure or
        repair as additional remaining work while ignoring future failures and
        competition from other jobs.
        """

        finish_offset = 0.0
        for op_id in range(self.job_op_counts[job_id]):
            op_idx = self.key_to_op[(job_id, op_id)]
            status = int(self.op_status[op_idx])
            if status == 2:
                continue
            if status == 1:
                machine = int(self.op_machine[op_idx])
                if machine >= 0:
                    finish_offset = max(
                        finish_offset,
                        self._machine_ready_delay(machine),
                    )
                    continue
            op = self._operation(op_idx)
            finish_offset = min(
                max(finish_offset, self._machine_ready_delay(machine))
                + float(processing_time)
                for machine, processing_time in op.machine_times.items()
            )
        return float(finish_offset)

    def _route_completion_lower_bound(
        self,
        job_id: int,
        *,
        include_active_repair: bool,
        forced_operation: int | None = None,
        forced_machine: int | None = None,
    ) -> float:
        """Observable route-completion lower bound under one optional commit.

        The two repair-contrast worlds share all current unfinished processing.
        Their only difference is whether repair time that is active *now* is
        retained.  Future failure windows are therefore never exposed.
        """

        if (forced_operation is None) != (forced_machine is None):
            raise ValueError("forced operation and machine must be supplied together")

        def ready_delay(machine: int) -> float:
            delay = self._machine_remaining_processing(int(machine))
            if include_active_repair:
                delay += self._machine_active_repair_remaining(int(machine))
            return float(delay)

        finish_offset = 0.0
        forced_seen = forced_operation is None
        for op_id in range(self.job_op_counts[job_id]):
            op_idx = self.key_to_op[(job_id, op_id)]
            status = int(self.op_status[op_idx])
            if status == 2:
                continue
            if status == 1:
                machine = int(self.op_machine[op_idx])
                if machine >= 0:
                    finish_offset = max(finish_offset, ready_delay(machine))
                    continue
            op = self._operation(op_idx)
            if forced_operation is not None and op_idx == int(forced_operation):
                machine = int(forced_machine)
                if machine not in op.machine_times:
                    raise ValueError("forced repair-contrast pair is ineligible")
                finish_offset = (
                    max(finish_offset, ready_delay(machine))
                    + float(op.machine_times[machine])
                )
                forced_seen = True
                continue
            finish_offset = min(
                max(finish_offset, ready_delay(machine))
                + float(processing_time)
                for machine, processing_time in op.machine_times.items()
            )
        if not forced_seen:
            raise ValueError("forced repair-contrast operation is not unfinished")
        return float(finish_offset)

    def _commit_repair_tardiness_contrast(
        self,
        job_id: int,
        operation: int,
        machine: int,
    ) -> float:
        """Pure active-repair TT contrast for a candidate pair afterstate."""

        with_repair = self._route_completion_lower_bound(
            int(job_id),
            include_active_repair=True,
            forced_operation=int(operation),
            forced_machine=int(machine),
        )
        without_repair = self._route_completion_lower_bound(
            int(job_id),
            include_active_repair=False,
            forced_operation=int(operation),
            forced_machine=int(machine),
        )
        due = self.instance.jobs[int(job_id)].due_date
        if due is None:
            return 0.0
        with_tt = max(0.0, float(self.time) + with_repair - float(due))
        without_tt = max(0.0, float(self.time) + without_repair - float(due))
        return float(max(0.0, with_tt - without_tt))

    def _status_baseline_remaining_work(self, job_id: int) -> float:
        """Availability-free residual lower bound from the observable state.

        Static operation features intentionally retain their historical suffix
        values.  This separate baseline is used only to isolate positive
        availability/status debt.  In particular, a running operation
        contributes its observable remaining processing rather than its full
        original duration, so an active repair cannot be cancelled by work
        already completed.
        """

        remaining = 0.0
        for op_id in range(self.job_op_counts[job_id]):
            op_idx = self.key_to_op[(job_id, op_id)]
            status = int(self.op_status[op_idx])
            if status == 2:
                continue
            if status == 1:
                machine = int(self.op_machine[op_idx])
                if machine >= 0:
                    remaining += self._machine_remaining_processing(machine)
                    continue
            remaining += self._operation(op_idx).min_time()
        return float(remaining)

    def _urgency_remaining_work(self, job_id: int) -> float:
        if self.config.urgency_mode == "availability_aware":
            return self._availability_aware_remaining_work(job_id)
        return self._remaining_min_work(job_id)

    def _remaining_ops_count(self, job_id: int, start_op_id: int = 0) -> int:
        remaining = 0
        for op_id in range(start_op_id, self.job_op_counts[job_id]):
            op_idx = self.key_to_op[(job_id, op_id)]
            remaining += int(self.op_status[op_idx] != 2)
        return remaining

    def _machine_features(self) -> np.ndarray:
        total_min_proc = self._observation_total_min_processing()
        scale = self._observation_scale()
        features = np.zeros((self.num_agents, self.machine_feature_dim), dtype=np.float32)
        for machine in range(self.num_agents):
            if self.config.separate_active_repair_feature:
                busy_remaining = self._machine_remaining_processing(machine)
                repair_remaining = self._machine_active_repair_remaining(machine)
            elif self.config.urgency_mode != "legacy_remaining_min":
                remaining = self._machine_ready_delay(machine)
            else:
                remaining = max(
                    0.0, float(self.machine_busy_until[machine] - self.time)
                )
            workload_feature = self.machine_workload[machine] / max(
                1.0, total_min_proc
            )
            energy_rate_feature = self.machine_energy_rates[machine] / max(
                1e-6, self.config.energy_rate_max
            )
            if self.config.feature_profile == "tardiness_only":
                workload_feature = 0.0
                energy_rate_feature = 0.0
            if self.config.separate_active_repair_feature:
                features[machine] = np.array(
                    [
                        1.0 if self._machine_available(machine) else 0.0,
                        busy_remaining / scale,
                        repair_remaining / scale,
                        workload_feature,
                        energy_rate_feature,
                    ],
                    dtype=np.float32,
                )
            else:
                features[machine] = np.array(
                    [
                        1.0 if self._machine_available(machine) else 0.0,
                        remaining / scale,
                        workload_feature,
                        energy_rate_feature,
                    ],
                    dtype=np.float32,
                )
        return features

    def _operation_features(self, agent_machine: int | None = None) -> np.ndarray:
        scale = self._observation_scale()
        ready_set = set(self._ready_operations())
        features = np.zeros((self.max_operations, self.op_feature_dim), dtype=np.float32)
        remaining_work = np.zeros(self.num_ops, dtype=np.float64)
        remaining_ops = np.zeros(self.num_ops, dtype=np.int32)
        separated_debt_by_job = np.zeros(
            self.instance.num_jobs,
            dtype=np.float64,
        )
        for job_id in range(self.instance.num_jobs):
            if self.config.hide_unarrived_jobs and not self._job_arrived(job_id):
                continue
            cumulative_work = 0.0
            cumulative_ops = 0
            for op_id in range(self.job_op_counts[job_id] - 1, -1, -1):
                op_idx = self.key_to_op[(job_id, op_id)]
                if self.op_status[op_idx] != 2:
                    cumulative_work += self._operation(op_idx).min_time()
                    cumulative_ops += 1
                remaining_work[op_idx] = cumulative_work
                remaining_ops[op_idx] = cumulative_ops
            if self.config.urgency_mode == "availability_aware":
                unfinished = [
                    self.key_to_op[(job_id, op_id)]
                    for op_id in range(self.job_op_counts[job_id])
                    if self.op_status[self.key_to_op[(job_id, op_id)]] != 2
                ]
                if unfinished:
                    first_unfinished = unfinished[0]
                    # Only the first unfinished operation can be ready for this
                    # job. Give that actionable node the exact current
                    # availability-aware route estimate; later context nodes keep
                    # their static suffix lower bounds until they become current.
                    remaining_work[first_unfinished] = (
                        self._availability_aware_remaining_work(job_id)
                    )
            elif self.config.urgency_mode == "separated_debt":
                separated_debt_by_job[job_id] = max(
                    0.0,
                    self._availability_aware_remaining_work(job_id)
                    - self._status_baseline_remaining_work(job_id),
                )
        for op_idx, (job_id, op_id) in enumerate(self.op_keys):
            if self.config.hide_unarrived_jobs and not self._job_arrived(job_id):
                continue
            job = self.instance.jobs[job_id]
            op = self._operation(op_idx)
            min_proc = op.min_time()
            due = job.due_date if job.due_date is not None else self.time + scale
            op_remaining_work = float(remaining_work[op_idx])
            op_remaining_ops = int(remaining_ops[op_idx])
            slack = due - self.time - op_remaining_work
            eligible = 0.0
            proc_for_agent = 0.0
            if agent_machine is not None and agent_machine in op.machine_times:
                eligible = 1.0
                proc_for_agent = op.machine_times[agent_machine] / scale
            repair_contrast = 0.0
            if (
                self.config.urgency_mode == "repair_contrast"
                and agent_machine is not None
                and op_idx in ready_set
                and agent_machine in op.machine_times
            ):
                repair_contrast = self._commit_repair_tardiness_contrast(
                    job_id,
                    op_idx,
                    int(agent_machine),
                )
            status = self.op_status[op_idx]
            op_features = [
                1.0,
                1.0 if op_idx in ready_set else 0.0,
                1.0 if status == 1 else 0.0,
                1.0 if status == 2 else 0.0,
                eligible,
                proc_for_agent,
                min_proc / scale,
                max(0.0, due - self.time) / scale,
                job.arrival_time / scale,
                op_id / max(1, len(job.operations) - 1),
            ]
            if self.op_feature_dim > 10:
                separated_debt = float(separated_debt_by_job[job_id])
                op_features.extend(
                    [
                        op_remaining_work / scale,
                        op_remaining_ops / max(1, self.job_op_counts[job_id]),
                        slack / scale,
                        max(0.0, -slack) / scale,
                        (
                            repair_contrast / scale
                            if (
                                self.config.feature_profile == "tardiness_only"
                                and self.config.urgency_mode == "repair_contrast"
                            )
                            else (
                                separated_debt / scale
                                if (
                                    self.config.feature_profile == "tardiness_only"
                                    and self.config.urgency_mode == "separated_debt"
                                )
                                else (
                                    0.0
                                    if self.config.feature_profile == "tardiness_only"
                                    else float(job.weight)
                                )
                            )
                        ),
                    ]
                )
            if self.op_feature_dim > 15:
                # Positive integer token used only to form permutation-invariant
                # same-job groups inside hierarchical actors. It is deliberately
                # excluded from numerical feature encoding.
                op_features.append(float(job_id + 1))
            features[op_idx, : min(len(op_features), self.op_feature_dim)] = np.asarray(
                op_features[: self.op_feature_dim],
                dtype=np.float32,
            )
        return features

    def _obs_state_mask(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        global_features = self._global_features()
        machine_features = self._machine_features()
        state_op_features = self._operation_features(agent_machine=None)
        eligibility = np.zeros((self.max_operations, self.num_agents), dtype=np.float32)
        proc_features = np.zeros_like(eligibility)
        scale = self._observation_scale()
        for op_idx in range(self.num_ops):
            job_id, _ = self.op_keys[op_idx]
            if self.config.hide_unarrived_jobs and not self._job_arrived(job_id):
                continue
            for machine, proc_time in self._operation(op_idx).machine_times.items():
                eligibility[op_idx, machine] = 1.0
                proc_features[op_idx, machine] = float(proc_time) / scale
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        for machine in range(self.num_agents):
            agent_one_hot = np.zeros(self.num_agents, dtype=np.float32)
            agent_one_hot[machine] = 1.0
            if self.config.urgency_mode == "repair_contrast":
                # This mode's channel 14 belongs to a candidate (machine,
                # operation) commit, so it must be generated per machine row.
                op_features = self._operation_features(agent_machine=machine)
            else:
                op_features = state_op_features.copy()
                op_features[:, 4] = eligibility[:, machine]
                op_features[:, 5] = proc_features[:, machine]
            obs[machine] = np.concatenate(
                [
                    global_features,
                    agent_one_hot,
                    machine_features.reshape(-1),
                    op_features.reshape(-1),
                ]
            )

        state = np.concatenate(
            [
                global_features,
                machine_features.reshape(-1),
                np.concatenate([state_op_features, eligibility], axis=1).reshape(-1),
            ]
        ).astype(np.float32)
        return obs, state, self.action_masks()

    def sample_valid_actions(self) -> np.ndarray:
        masks = self.action_masks()
        actions = np.zeros(self.num_agents, dtype=np.int64)
        for agent in range(self.num_agents):
            valid = np.flatnonzero(masks[agent])
            actions[agent] = int(self.rng.choice(valid))
        return actions


class StaticBenchmarkFJSPEnv(BaseFJSPEnv):
    """Static public-benchmark style environment."""

    def _make_instance(self) -> FJSPInstance:
        if self.config.instance_path:
            if Path(self.config.instance_path).suffix.lower() == ".json":
                return load_instance(self.config.instance_path)
            return load_benchmark_instance(
                self.config.instance_path,
                due_date_factor=self.config.due_date_factor,
            )
        return load_builtin_instance(self.config.builtin_instance)


class RandomFJSPEnv(BaseFJSPEnv):
    """Synthetic static FJSP instances, regenerated at every reset."""

    def _make_instance(self) -> FJSPInstance:
        seed = int(self.rng.integers(0, 2**31 - 1))
        return generate_random_instance(
            num_jobs=self.config.num_jobs,
            num_machines=self.config.num_machines,
            ops_per_job_range=(
                self.config.ops_per_job_min,
                self.config.ops_per_job_max,
            ),
            eligible_machine_range=(
                self.config.eligible_machines_min,
                self.config.eligible_machines_max,
            ),
            processing_time_range=(
                self.config.processing_time_min,
                self.config.processing_time_max,
            ),
            arrival_mode=self.config.arrival_mode,
            arrival_rate=self.config.arrival_rate,
            due_date_factor=self.config.due_date_factor,
            due_date_noise=self.config.due_date_noise,
            seed=seed,
            name="random_static_fjsp",
        )


class DynamicMultiObjectiveFJSPEnv(RandomFJSPEnv):
    """Dynamic FJSP with arrivals, machine failures, due dates, and energy."""

    def _make_instance(self) -> FJSPInstance:
        seed = int(self.rng.integers(0, 2**31 - 1))
        return generate_random_instance(
            num_jobs=self.config.num_jobs,
            num_machines=self.config.num_machines,
            ops_per_job_range=(
                self.config.ops_per_job_min,
                self.config.ops_per_job_max,
            ),
            eligible_machine_range=(
                self.config.eligible_machines_min,
                self.config.eligible_machines_max,
            ),
            processing_time_range=(
                self.config.processing_time_min,
                self.config.processing_time_max,
            ),
            arrival_mode=self.config.dynamic_arrival_mode,
            arrival_rate=self.config.arrival_rate,
            due_date_factor=self.config.due_date_factor,
            due_date_noise=self.config.due_date_noise,
            seed=seed,
            name="dynamic_multiobjective_fjsp",
        )


class DynamicBenchmarkFJSPEnv(StaticBenchmarkFJSPEnv):
    """Public FJSP routes/times with reproducible dynamic disturbances."""

    def _load_base_instance(self) -> FJSPInstance:
        if not self.config.instance_path:
            raise ValueError("dynamic_benchmark requires env.instance_path")
        path = Path(self.config.instance_path)
        if path.suffix.lower() == ".json":
            return load_instance(path)
        return load_benchmark_instance(path, due_date_factor=None)

    def _make_instance(self) -> FJSPInstance:
        base_instance = self._load_base_instance()
        dynamic_seed = int(self.rng.integers(0, 2**31 - 1))
        return inject_dynamic_attributes(
            base_instance,
            arrival_mode=self.config.dynamic_arrival_mode,
            arrival_rate=self.config.arrival_rate,
            due_date_factor=self.config.due_date_factor,
            due_date_noise=self.config.due_date_noise,
            seed=dynamic_seed,
        )


def make_fjsp_env(config: FJSPEnvConfig | dict[str, Any]) -> BaseFJSPEnv:
    if isinstance(config, dict):
        config = FJSPEnvConfig(**config)
    env_type = config.env_type.lower()
    if env_type == "static":
        return StaticBenchmarkFJSPEnv(config)
    if env_type == "random":
        return RandomFJSPEnv(config)
    if env_type in {"dynamic", "dynamic_multiobjective"}:
        return DynamicMultiObjectiveFJSPEnv(config)
    if env_type in {"dynamic_benchmark", "benchmark_dynamic"}:
        return DynamicBenchmarkFJSPEnv(config)
    raise ValueError(f"Unsupported env_type: {config.env_type}")
