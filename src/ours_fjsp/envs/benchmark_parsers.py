from __future__ import annotations

from pathlib import Path
import re

from .instances import FJSPInstance, JobSpec, OperationSpec, save_instance


COMMENT_PREFIXES = ("#", "%", "//")


def _strip_comment(line: str) -> str:
    clean = line
    for prefix in COMMENT_PREFIXES:
        if prefix in clean:
            clean = clean.split(prefix, 1)[0]
    return clean.strip()


def _numeric_rows(path: str | Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw_line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue
        row: list[float] = []
        for part in re.split(r"[\s,;]+", line):
            if not part:
                continue
            if part in {"-", "*", "NA", "NaN", "nan"}:
                row.append(float("inf"))
            else:
                row.append(float(part))
        if row:
            rows.append(row)
    return rows


def parse_fjs_file(
    path: str | Path,
    *,
    name: str | None = None,
    machine_index_base: int | str = "auto",
    due_date_factor: float | None = None,
) -> FJSPInstance:
    """Parse the common Brandimarte/Hurink `.fjs`-style FJSP format.

    Expected numeric layout:

    ```
    <num_jobs> <num_machines> [optional metadata...]
    <num_ops_job_0>
      <num_eligible_machines_op_0> <machine> <proc_time> ...
      <num_eligible_machines_op_1> <machine> <proc_time> ...
    ...
    ```

    Many public Brandimarte, Hurink, and Kacem copies use this compact
    whitespace-only format. Machine IDs are commonly 1-based; `auto` subtracts
    one when the file contains no machine 0 and all IDs are within 1..m.
    """

    path = Path(path)
    rows = _numeric_rows(path)
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"File does not contain a valid FJSP header: {path}")
    num_jobs = int(rows[0][0])
    num_machines = int(rows[0][1])
    # Public .fjs files commonly add average machine flexibility as a third
    # header value. It is metadata, not the first job's operation count.
    tokens = [token for row in rows[1:] for token in row]
    cursor = 0
    jobs: list[JobSpec] = []
    raw_machine_ids: list[int] = []

    for job_id in range(num_jobs):
        if cursor >= len(tokens):
            raise ValueError(f"Unexpected end while reading job {job_id} in {path}")
        num_ops = int(tokens[cursor])
        cursor += 1
        operations: list[OperationSpec] = []
        for op_id in range(num_ops):
            if cursor >= len(tokens):
                raise ValueError(
                    f"Unexpected end while reading job {job_id}, operation {op_id} in {path}"
                )
            num_alternatives = int(tokens[cursor])
            cursor += 1
            machine_times: dict[int, float] = {}
            for _ in range(num_alternatives):
                if cursor + 1 >= len(tokens):
                    raise ValueError(
                        f"Unexpected end in machine-time pairs for job {job_id}, "
                        f"operation {op_id} in {path}"
                    )
                machine = int(tokens[cursor])
                proc_time = float(tokens[cursor + 1])
                cursor += 2
                raw_machine_ids.append(machine)
                machine_times[machine] = proc_time
            operations.append(OperationSpec(machine_times=machine_times))
        jobs.append(JobSpec(operations=operations))

    if cursor != len(tokens):
        trailing = tokens[cursor:]
        raise ValueError(
            f"Unexpected trailing numeric data after {num_jobs} jobs in {path}: "
            f"{len(trailing)} token(s)"
        )

    if machine_index_base == "auto":
        if raw_machine_ids and min(raw_machine_ids) >= 1 and max(raw_machine_ids) <= num_machines:
            base = 1
        else:
            base = 0
    else:
        base = int(machine_index_base)

    normalized_jobs: list[JobSpec] = []
    for job in jobs:
        operations = []
        min_total = 0.0
        for op in job.operations:
            normalized = {machine - base: time for machine, time in op.machine_times.items()}
            if min(normalized) < 0 or max(normalized) >= num_machines:
                raise ValueError(
                    f"Machine index out of range after base={base} normalization in {path}"
                )
            operations.append(OperationSpec(normalized))
            min_total += min(normalized.values())
        due_date = None if due_date_factor is None else due_date_factor * min_total
        normalized_jobs.append(JobSpec(operations=operations, due_date=due_date))

    return FJSPInstance(
        jobs=normalized_jobs,
        num_machines=num_machines,
        name=name or path.stem,
        metadata={
            "source_path": str(path),
            "parser": "fjs_compact",
            "machine_index_base": base,
        },
    )


def parse_brandimarte(path: str | Path, **kwargs) -> FJSPInstance:
    return parse_fjs_file(path, name=kwargs.pop("name", Path(path).stem), **kwargs)


def parse_hurink(path: str | Path, **kwargs) -> FJSPInstance:
    return parse_fjs_file(path, name=kwargs.pop("name", Path(path).stem), **kwargs)


def parse_kacem(path: str | Path, **kwargs) -> FJSPInstance:
    return parse_fjs_file(path, name=kwargs.pop("name", Path(path).stem), **kwargs)


def load_benchmark_instance(
    path: str | Path,
    *,
    benchmark: str = "auto",
    machine_index_base: int | str = "auto",
    due_date_factor: float | None = None,
) -> FJSPInstance:
    benchmark_key = benchmark.lower()
    parser_map = {
        "auto": parse_fjs_file,
        "fjs": parse_fjs_file,
        "brandimarte": parse_brandimarte,
        "mk": parse_brandimarte,
        "hurink": parse_hurink,
        "kacem": parse_kacem,
    }
    if benchmark_key not in parser_map:
        raise ValueError(f"Unsupported benchmark family: {benchmark}")
    return parser_map[benchmark_key](
        path,
        machine_index_base=machine_index_base,
        due_date_factor=due_date_factor,
    )


def convert_benchmark_to_json(
    input_path: str | Path,
    output_path: str | Path,
    *,
    benchmark: str = "auto",
    machine_index_base: int | str = "auto",
    due_date_factor: float | None = None,
) -> FJSPInstance:
    instance = load_benchmark_instance(
        input_path,
        benchmark=benchmark,
        machine_index_base=machine_index_base,
        due_date_factor=due_date_factor,
    )
    save_instance(instance, output_path)
    return instance
