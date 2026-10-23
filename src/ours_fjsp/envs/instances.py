from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np


@dataclass(frozen=True)
class OperationSpec:
    """One operation with alternative machine processing times."""

    machine_times: dict[int, float]

    def min_time(self) -> float:
        return min(self.machine_times.values())

    def machines(self) -> tuple[int, ...]:
        return tuple(sorted(self.machine_times))


@dataclass(frozen=True)
class JobSpec:
    operations: list[OperationSpec]
    arrival_time: float = 0.0
    due_date: float | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class FJSPInstance:
    jobs: list[JobSpec]
    num_machines: int
    name: str = "fjsp_instance"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    @property
    def num_operations(self) -> int:
        return sum(len(job.operations) for job in self.jobs)

    @property
    def max_ops_per_job(self) -> int:
        return max((len(job.operations) for job in self.jobs), default=0)

    def operation_keys(self) -> list[tuple[int, int]]:
        keys: list[tuple[int, int]] = []
        for job_id, job in enumerate(self.jobs):
            for op_id in range(len(job.operations)):
                keys.append((job_id, op_id))
        return keys

    def min_total_processing_time(self, job_id: int) -> float:
        return sum(op.min_time() for op in self.jobs[job_id].operations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_machines": self.num_machines,
            "metadata": self.metadata,
            "jobs": [
                {
                    "arrival_time": job.arrival_time,
                    "due_date": job.due_date,
                    "weight": job.weight,
                    "operations": [
                        {"machine_times": {str(k): v for k, v in op.machine_times.items()}}
                        for op in job.operations
                    ],
                }
                for job in self.jobs
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FJSPInstance":
        jobs: list[JobSpec] = []
        for job_data in data["jobs"]:
            operations = [
                OperationSpec({int(k): float(v) for k, v in op["machine_times"].items()})
                for op in job_data["operations"]
            ]
            jobs.append(
                JobSpec(
                    operations=operations,
                    arrival_time=float(job_data.get("arrival_time", 0.0)),
                    due_date=(
                        None
                        if job_data.get("due_date") is None
                        else float(job_data["due_date"])
                    ),
                    weight=float(job_data.get("weight", 1.0)),
                )
            )
        return cls(
            jobs=jobs,
            num_machines=int(data["num_machines"]),
            name=data.get("name", "fjsp_instance"),
            metadata=data.get("metadata", {}),
        )


def save_instance(instance: FJSPInstance, path: str | Path) -> None:
    Path(path).write_text(json.dumps(instance.to_dict(), indent=2), encoding="utf-8")


def load_instance(path: str | Path) -> FJSPInstance:
    return FJSPInstance.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _sample_operation(
    rng: np.random.Generator,
    num_machines: int,
    eligible_machine_range: tuple[int, int],
    processing_time_range: tuple[int, int],
) -> OperationSpec:
    low, high = eligible_machine_range
    eligible_count = int(rng.integers(low, high + 1))
    eligible_count = max(1, min(num_machines, eligible_count))
    machines = rng.choice(num_machines, size=eligible_count, replace=False)
    p_low, p_high = processing_time_range
    machine_times = {
        int(machine): float(rng.integers(p_low, p_high + 1)) for machine in machines
    }
    return OperationSpec(machine_times=machine_times)


def generate_random_instance(
    *,
    num_jobs: int,
    num_machines: int,
    ops_per_job_range: tuple[int, int] = (3, 6),
    eligible_machine_range: tuple[int, int] = (1, 3),
    processing_time_range: tuple[int, int] = (1, 20),
    arrival_mode: str = "zero",
    arrival_rate: float = 0.25,
    due_date_factor: float = 1.6,
    due_date_noise: float = 0.15,
    seed: int | None = None,
    name: str = "random_fjsp",
) -> FJSPInstance:
    """Generate a synthetic FJSP instance.

    arrival_mode:
        zero: all jobs are available at t=0.
        uniform: release times are uniformly sampled.
        poisson: interarrival times are exponential with rate arrival_rate.
    """

    rng = np.random.default_rng(seed)
    jobs: list[JobSpec] = []
    current_arrival = 0.0
    max_min_job_time = 0.0
    sampled_ops: list[list[OperationSpec]] = []

    for _ in range(num_jobs):
        num_ops = int(rng.integers(ops_per_job_range[0], ops_per_job_range[1] + 1))
        operations = [
            _sample_operation(
                rng,
                num_machines,
                eligible_machine_range,
                processing_time_range,
            )
            for _ in range(num_ops)
        ]
        sampled_ops.append(operations)
        max_min_job_time = max(max_min_job_time, sum(op.min_time() for op in operations))

    if arrival_mode == "zero":
        arrivals = [0.0] * num_jobs
    elif arrival_mode == "uniform":
        horizon = max(1.0, num_jobs * max_min_job_time / max(1, num_machines))
        arrivals = sorted(float(x) for x in rng.uniform(0.0, 0.4 * horizon, num_jobs))
    elif arrival_mode == "poisson":
        arrivals = []
        for _ in range(num_jobs):
            current_arrival += float(rng.exponential(1.0 / max(arrival_rate, 1e-6)))
            arrivals.append(current_arrival)
    else:
        raise ValueError(f"Unsupported arrival_mode: {arrival_mode}")

    for job_id, operations in enumerate(sampled_ops):
        min_proc = sum(op.min_time() for op in operations)
        noise = 1.0 + float(rng.uniform(-due_date_noise, due_date_noise))
        due_date = arrivals[job_id] + due_date_factor * min_proc * noise
        jobs.append(
            JobSpec(
                operations=operations,
                arrival_time=float(arrivals[job_id]),
                due_date=float(max(arrivals[job_id] + min_proc, due_date)),
            )
        )

    return FJSPInstance(
        jobs=jobs,
        num_machines=num_machines,
        name=name,
        metadata={
            "generator": "synthetic",
            "arrival_mode": arrival_mode,
            "due_date_factor": due_date_factor,
        },
    )


def inject_dynamic_attributes(
    instance: FJSPInstance,
    *,
    arrival_mode: str = "poisson",
    arrival_rate: float = 0.35,
    due_date_factor: float = 1.7,
    due_date_noise: float = 0.25,
    seed: int | None = None,
) -> FJSPInstance:
    """Add release times and due dates without changing benchmark process data.

    The returned instance keeps every operation, eligible machine, and
    processing time from ``instance``. Only dynamic attributes are sampled.
    Using the same seed therefore gives every evaluated policy exactly the
    same arrivals and due dates.
    """

    rng = np.random.default_rng(seed)
    num_jobs = instance.num_jobs
    if arrival_mode == "zero":
        arrivals = np.zeros(num_jobs, dtype=float)
    elif arrival_mode == "uniform":
        total_min_work = sum(instance.min_total_processing_time(j) for j in range(num_jobs))
        horizon = total_min_work / max(1, instance.num_machines)
        arrivals = np.sort(rng.uniform(0.0, 0.4 * max(1.0, horizon), num_jobs))
    elif arrival_mode == "poisson":
        interarrivals = rng.exponential(
            1.0 / max(arrival_rate, 1e-6),
            size=num_jobs,
        )
        arrivals = np.cumsum(interarrivals)
    else:
        raise ValueError(f"Unsupported arrival_mode: {arrival_mode}")

    jobs: list[JobSpec] = []
    for job_id, source_job in enumerate(instance.jobs):
        min_proc = instance.min_total_processing_time(job_id)
        noise = 1.0 + float(rng.uniform(-due_date_noise, due_date_noise))
        due_date = float(arrivals[job_id] + due_date_factor * min_proc * noise)
        jobs.append(
            JobSpec(
                operations=source_job.operations,
                arrival_time=float(arrivals[job_id]),
                due_date=max(float(arrivals[job_id] + min_proc), due_date),
                weight=source_job.weight,
            )
        )

    metadata = dict(instance.metadata)
    metadata.update(
        {
            "dynamic_overlay": "release_dates_and_due_dates",
            "arrival_mode": arrival_mode,
            "arrival_rate": float(arrival_rate),
            "due_date_factor": float(due_date_factor),
            "due_date_noise": float(due_date_noise),
            "dynamic_seed": seed,
            "benchmark_routes_and_processing_times_preserved": True,
        }
    )
    return FJSPInstance(
        jobs=jobs,
        num_machines=instance.num_machines,
        name=f"{instance.name}_dynamic",
        metadata=metadata,
    )


def load_builtin_instance(name: str = "toy_static") -> FJSPInstance:
    """Small built-in instances for smoke tests and algorithm debugging."""

    if name != "toy_static":
        raise ValueError(f"Unknown built-in instance: {name}")
    jobs = [
        JobSpec(
            [
                OperationSpec({0: 3, 1: 4}),
                OperationSpec({1: 2, 2: 5}),
                OperationSpec({0: 4, 2: 3}),
            ],
            due_date=14,
        ),
        JobSpec(
            [
                OperationSpec({0: 2, 2: 4}),
                OperationSpec({0: 3, 1: 3}),
                OperationSpec({1: 5, 2: 2}),
            ],
            due_date=13,
        ),
        JobSpec(
            [
                OperationSpec({1: 4, 2: 3}),
                OperationSpec({0: 5, 2: 3}),
            ],
            due_date=11,
        ),
        JobSpec(
            [
                OperationSpec({0: 4, 1: 2}),
                OperationSpec({1: 4, 2: 4}),
                OperationSpec({0: 3, 2: 5}),
            ],
            due_date=15,
        ),
    ]
    return FJSPInstance(
        jobs=jobs,
        num_machines=3,
        name="toy_static",
        metadata={"source": "built-in smoke-test instance"},
    )
