"""Launch the main method over selected scenarios and training seeds."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .protocol import TRAIN_SEEDS
from .scenarios import SCENARIOS


def commands_for(args, scenario: str, seed: int) -> tuple[list[str], list[str]]:
    job = args.output.resolve() / "training" / f"{scenario}_s{seed}"
    train = [sys.executable, "-m", "ours_fjsp.train", "--scenario", scenario,
             "--seed", str(seed), "--output", str(job)]
    if args.config_dir is not None:
        config = args.config_dir / f"{scenario}.yaml"
        if not config.is_file():
            raise FileNotFoundError(config)
        train += ["--config", str(config.resolve())]
    if args.resume and (job / "latest_state.pt").is_file():
        train += ["--resume"]
    evaluation = [sys.executable, "-m", "ours_fjsp.evaluate", "--scenario", scenario,
                  "--checkpoint", str(job / "checkpoint.pt"), "--episodes", "12",
                  "--output", str(args.output.resolve() / "evaluations" / f"{scenario}_s{seed}.json")]
    instance = getattr(args, "mk08_instance" if scenario == "Mk08" else "mk09_instance", None)
    if scenario in ("Mk08", "Mk09"):
        if instance is None or not instance.is_file():
            raise ValueError(f"{scenario} requires --{scenario.lower()}-instance with a local benchmark file")
        train += ["--instance", str(instance.resolve())]
        evaluation += ["--instance", str(instance.resolve())]
    return train, evaluation


def run_job(commands: tuple[list[str], list[str]]) -> None:
    for command in commands:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=TRAIN_SEEDS, default=list(TRAIN_SEEDS))
    parser.add_argument("--config-dir", type=Path, help="Optional directory containing the five scenario YAMLs")
    parser.add_argument("--mk08-instance", type=Path)
    parser.add_argument("--mk09-instance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="Independent processes; one Torch CPU thread per process")
    parser.add_argument("--resume", action="store_true", help="Resume existing full training states; new jobs start from scratch")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating outputs or starting training")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if len(set(args.scenarios)) != len(args.scenarios) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate scenarios or seeds would target the same output")
    jobs = [commands_for(args, case, seed) for case in args.scenarios for seed in args.seeds]
    if args.dry_run:
        print(json.dumps({"models": len(jobs), "evaluations_per_model": 12,
                          "commands": [[shlex.join(command) for command in job] for job in jobs]}, indent=2))
        return
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        for _ in executor.map(run_job, jobs):
            pass


if __name__ == "__main__":
    main()
