# Matching-MAPPO

Code for **Matching-MAPPO: Urgency-Aware Joint Matching for Dynamic Flexible Job Shop Scheduling**.

This repository contains the main Matching-MAPPO method: the dynamic scheduling environment, urgency features, graph actor, centralized critic, exact joint partial-matching policy, PPO training, and evaluation. [中文说明](README_zh.md).

## Installation

Python 3.9 or newer is required. The training output lock uses POSIX file locking; use Linux, macOS, or Windows with WSL.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

Runtime dependencies are NumPy, PyTorch, and PyYAML. The reference CPU environment uses Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2, and PyYAML 6.0.3. Exact numerical replay depends on the runtime and hardware; it is not promised across versions or accelerators.

## Main experiment

The five configurations are in `configs/`. All use the complete method: observable arrivals and machine states, the paper's remaining-processing/slack features, cumulative machine load, bidirectional graph messages, learned wait scores, exact nonempty legal partial matchings, and pure negative incremental total tardiness.

Training follows the paper's **minimum-1200-natural-episode protocol**:

1. Complete the original 300 updates of 512 decisions without stopping early at episode 1200.
2. If that prefix already completed at least 1200 natural episodes, retain its actual final episode count.
3. Otherwise continue the same agent, optimizer, environment, and random streams to the first natural episode 1200. Update the final, possibly short rollout using its actual length and zero terminal bootstrap.

Truncations do not count toward the natural-episode target, and no artificial unfinished-job penalty is added. Episode, decision, update, and truncation counts are recorded separately. Thus different scenes can have different actual episode counts and decision budgets.

### Train one model

```bash
matching-mappo-train --config configs/S20L.yaml --output outputs/S20L_s2400001
```

The independent training seeds are `2400001`, `2400002`, and `2400003`. Each uses its original disjoint third of the scenario's 10,000-instance training pool. Override the seed with `--seed`.

A complete state is saved before the first rollout and after every PPO update. Resume the same run with unchanged source, runtime, configuration, and budget:

```bash
matching-mappo-train --config configs/S20L.yaml --output outputs/S20L_s2400001 --resume
```

`checkpoint.pt` is the final inference checkpoint. `latest_state.pt` contains the full optimizer, environment, random generators, counters, and training history; it is intended only for resuming states generated locally by this code. A nonempty output is rejected unless `--resume` is explicit. Historical paper checkpoints are not required.

### Evaluate one model

```bash
matching-mappo-evaluate --scenario S20L \
  --checkpoint outputs/S20L_s2400001/checkpoint.pt \
  --output outputs/evaluation_S20L_s2400001.json
```

Evaluation uses deterministic maximum-score legal matching and defaults to **12 common physical instances**. The seed starts are `3600000`, `3601000`, `3602000`, `3603000`, and `3604000` for S20L, S20H, S40H, Mk08, and Mk09. Each row includes TT, makespan, mean flow time, completion information, the actual return, and physical/checkpoint fingerprints. An unfinished schedule retains its observed return and has null final metrics.

### Mk08 and Mk09 inputs

The S scenarios generate their instances. Supply the original Brandimarte problem files for Mk08/Mk09:

```bash
matching-mappo-train --config configs/Mk08.yaml --instance data/Mk08.fjs \
  --output outputs/Mk08_s2400001
matching-mappo-evaluate --scenario Mk08 --instance data/Mk08.fjs \
  --checkpoint outputs/Mk08_s2400001/checkpoint.pt \
  --output outputs/evaluation_Mk08_s2400001.json
```

Benchmark data are external inputs, not bundled outputs. The main-study SHA256 fingerprints are:

| Input | SHA256                                                       |
| ----- | ------------------------------------------------------------ |
| Mk08  | `d06776e82c0fe7b00a705bda06759ad4668025fefd1125f6639fd62d9a8bbf18` |
| Mk09  | `93a7f6dc3ff60d87b88511f7f0a73ce7116f9e1399822176bb8bd0adf51b80eb` |

Other compatible inputs can be used, but are marked as custom runs rather than the reference paper protocol.

### Run the five-scene suite

This command trains the full **5 scenes × 3 seeds** and evaluates each final model on its 12 instances. It does not select favorable seeds. The paper's post hoc S40H display selection is a reporting choice, not a training filter.

```bash
matching-mappo-experiment --config-dir configs \
  --mk08-instance data/Mk08.fjs --mk09-instance data/Mk09.fjs \
  --workers 1 --output outputs/main
```

Choose independent process concurrency with `--workers`. Add `--resume` to continue existing full states. Preview a smaller suite without creating outputs:

```bash
matching-mappo-experiment --scenarios S20L S20H S40H \
  --config-dir configs --output outputs/preview --dry-run
```

### Short verification

```bash
matching-mappo-train --scenario S20L --smoke --updates 2 \
  --rollout-steps 8 --output outputs/smoke
python -m pytest -q
```

Smoke and custom budgets are explicitly marked in metadata. They are not formal paper experiments. No experiment starts during installation or import.

## Repository layout

```text
configs/                 Five main-study configurations
src/ours_fjsp/
  algorithms/            Graph actor, critic, exact policy, PPO, initialization
  envs/                  Physical environment and problem parsers
  train.py               Training, budget gate, complete-state resume
  evaluate.py            Common-instance evaluation
  experiment.py          Multi-scene/multi-seed launcher
tests/                   Formula, feasibility, policy, budget and resume checks
```

The repository includes no trained weights, experiment results, comparison methods, or component-removal experiments. Generated weights, logs, reports, caches, and local datasets are excluded by `.gitignore`.
