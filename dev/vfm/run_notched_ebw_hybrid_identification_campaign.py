"""Run a resumable manifest-defined notched-EBW identification campaign.

The round manifest contains a ``cases`` list. Each case requires ``name`` and
``arguments`` (a list of arguments accepted by the bivariate caller).  This
keeps Rounds 2 and 3 on one launcher while allowing Round 1 to freeze the exact
objective arguments before expensive online runs begin.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pyvale.vfm.campaignprogress import ProgressEstimate, atomic_write_json


@dataclass(slots=True)
class CaseResult:
    name: str
    status: str
    return_code: int
    runtime_seconds: float
    result_file: str
    log_file: str


def main() -> None:
    args = _parse_args()
    config = json.loads(args.round_manifest.read_text(encoding="utf-8"))
    cases = config.get("cases", [])
    if not cases or any("name" not in case or "arguments" not in case for case in cases):
        raise ValueError("Round manifest requires cases with name and arguments.")
    dataset = args.dataset.expanduser().resolve()
    input_dir = dataset / "prepared"
    if not (input_dir / "experiment_data.yaml").is_file():
        raise FileNotFoundError(f"Prepared input is absent below {input_dir}.")
    output_root = (
        dataset / "identification"
        if args.output_root is None
        else args.output_root.expanduser().resolve()
    )
    root = output_root / "prepared" / args.campaign_name
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "campaign_manifest.json"
    results: list[CaseResult] = []
    started = time.monotonic()
    _write_manifest(manifest_path, args, config, results)
    print(
        f"campaign={args.campaign_name} cases={len(cases)} jobs={args.jobs} "
        f"parallel_workers={args.parallel_workers}", flush=True
    )
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        pending = {
            executor.submit(_run_case, case, args, dataset, output_root, logs): case
            for case in cases
        }
        last_heartbeat = started
        while pending:
            done, pending = wait(
                pending,
                timeout=args.progress_interval_seconds or None,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                result = future.result()
                results.append(result)
                print(
                    f"case {result.status} name={result.name} "
                    f"runtime={result.runtime_seconds / 60.0:.1f}min "
                    f"log={result.log_file}", flush=True
                )
                _write_manifest(manifest_path, args, config, results)
            now = time.monotonic()
            if pending and args.progress_interval_seconds and now - last_heartbeat >= args.progress_interval_seconds:
                status_counts = {
                    status: sum(item.status == status for item in results)
                    for status in ("complete", "failed", "skipped")
                }
                active = sum(future.running() for future in pending)
                estimate = ProgressEstimate.from_counts(len(results), len(cases), started)
                print(
                    estimate.line()
                    + f" active={active} queued={len(pending) - active}"
                    + " ".join(f" {key}={value}" for key, value in status_counts.items()),
                    flush=True,
                )
                last_heartbeat = now
    failures = [result for result in results if result.status == "failed"]
    print(f"manifest={manifest_path}", flush=True)
    if failures:
        raise SystemExit(1)


def _run_case(case, args, dataset, output_root, logs) -> CaseResult:
    name = str(case["name"])
    run_name = f"prepared/{args.campaign_name}/{name}"
    result_file = output_root / run_name / "identification_result.yaml"
    log_file = logs / f"{name}.log"
    if result_file.is_file():
        return CaseResult(name, "skipped", 0, 0.0, str(result_file), str(log_file))
    command = [
        "uv", "run", "--no-sync", "python",
        "dev/vfm/call_notched_ebw_bivariate_identification.py",
        "--input", str(dataset / "prepared"),
        "--output-root", str(output_root),
        "--run-name", run_name,
        "--parallel-workers", str(args.parallel_workers),
        "--no-progress",
        *[str(value) for value in case["arguments"]],
    ]
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/pyvale-matplotlib",
        "UV_CACHE_DIR": "/tmp/pyvale-uv-cache",
    })
    began = time.monotonic()
    with log_file.open("w", encoding="utf-8") as stream:
        stream.write(f"command={json.dumps(command)}\n")
        stream.flush()
        completed = subprocess.run(
            command, cwd=args.pyvale_root, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    runtime = time.monotonic() - began
    status = "complete" if completed.returncode == 0 and result_file.is_file() else "failed"
    return CaseResult(name, status, completed.returncode, runtime, str(result_file), str(log_file))


def _write_manifest(path, args, config, results):
    completed = {result.name: result for result in results}
    cases = []
    for case in config["cases"]:
        record = dict(case)
        record.update(asdict(completed[case["name"]]) if case["name"] in completed else {"status": "pending"})
        cases.append(record)
    atomic_write_json(path, {
        "campaign_name": args.campaign_name,
        "round_manifest": str(args.round_manifest),
        "configuration": {
            "jobs": args.jobs, "parallel_workers": args.parallel_workers,
            "progress_interval_seconds": args.progress_interval_seconds,
        },
        "cases": cases,
    })


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Identification output root; defaults to DATASET/identification.",
    )
    parser.add_argument("--campaign-name", required=True)
    parser.add_argument("--pyvale-root", type=Path, default=Path.cwd())
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--progress-interval-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.jobs < 1 or args.parallel_workers < 1:
        parser.error("--jobs and --parallel-workers must be positive")
    if args.progress_interval_seconds < 0.0:
        parser.error("--progress-interval-seconds must be non-negative")
    args.round_manifest = args.round_manifest.expanduser().resolve()
    args.pyvale_root = args.pyvale_root.expanduser().resolve()
    return args


if __name__ == "__main__":
    main()
