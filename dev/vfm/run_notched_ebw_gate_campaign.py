"""Run the notched-EBW gate and objective-discrimination campaign.

The campaign is resumable and deliberately runs independent identifications
as separate processes.  The default matrix contains four acceptance gates
across eight optimiser seeds for SPD+sensitivity growth, plus a matched
SPD+EGI control at the provisional one-percent gate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_DATASET = Path.home() / (
    "projects/pyvale-vfm-test-data/notched-ebw/synthetic-fe/"
    "wdbn1-idealised-yield/pyvale-vfm"
)


@dataclass(frozen=True, slots=True)
class CampaignCase:
    name: str
    policy: str
    gate: float
    seed: int


@dataclass(slots=True)
class CaseResult:
    case: CampaignCase
    status: str
    return_code: int
    runtime_seconds: float
    result_file: str
    log_file: str


def main() -> None:
    args = _parse_args()
    dataset = args.dataset.expanduser().resolve()
    input_dir = dataset / "prepared"
    if not (input_dir / "experiment_data.yaml").is_file():
        raise FileNotFoundError(
            f"Prepared experiment not found below {input_dir}. "
            "Pass --dataset explicitly if the workstation path differs."
        )

    campaign_root = dataset / "identification" / "prepared" / args.campaign_name
    log_root = campaign_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    cases = _build_cases(args.gates, args.seeds, args.control_gate)
    manifest_path = campaign_root / "campaign_manifest.json"
    _write_manifest(manifest_path, args, dataset, cases, [])

    print(
        f"Running {len(cases)} cases with {args.jobs} concurrent processes "
        f"and {args.parallel_workers} objective workers per process."
    )
    results: list[CaseResult] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(_run_case, case, args, dataset, log_root): case
            for case in cases
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result.status:7s} {result.case.name} "
                f"({result.runtime_seconds / 60.0:.1f} min)"
            )
            _write_manifest(manifest_path, args, dataset, cases, results)

    failures = [item for item in results if item.return_code != 0]
    print(f"Manifest: {manifest_path}")
    if failures:
        print("Failed cases:", file=sys.stderr)
        for item in failures:
            print(f"  {item.case.name}: {item.log_file}", file=sys.stderr)
        raise SystemExit(1)


def _build_cases(
    gates: tuple[float, ...],
    seeds: tuple[int, ...],
    control_gate: float,
) -> list[CampaignCase]:
    cases = [
        CampaignCase(
            name=f"spd_sensitivity_gate{_gate_label(gate)}_seed{seed:02d}",
            policy="sensitivity_correction",
            gate=gate,
            seed=seed,
        )
        for gate in gates
        for seed in seeds
    ]
    cases.extend(
        CampaignCase(
            name=f"spd_egi_gate{_gate_label(control_gate)}_seed{seed:02d}",
            policy="egi_peak",
            gate=control_gate,
            seed=seed,
        )
        for seed in seeds
    )
    return cases


def _gate_label(gate: float) -> str:
    return f"{100.0 * gate:.1f}pct".replace(".", "p")


def _run_case(
    case: CampaignCase,
    args: argparse.Namespace,
    dataset: Path,
    log_root: Path,
) -> CaseResult:
    run_name = f"prepared/{args.campaign_name}/{case.name}"
    result_file = dataset / "identification" / run_name / "identification_result.yaml"
    log_file = log_root / f"{case.name}.log"
    if result_file.is_file():
        return CaseResult(case, "skipped", 0, 0.0, str(result_file), str(log_file))

    command = [
        "uv", "run", "--no-sync", "python",
        "dev/vfm/call_notched_ebw_bivariate_identification.py",
        "--input", str(dataset / "prepared"),
        "--output-root", str(dataset / "identification"),
        "--run-name", run_name,
        "--kernel-type", "bivariate_spd",
        "--basis-growth-policy", case.policy,
        "--minimum-objective-improvement", str(case.gate),
        "--max-basis-functions", str(args.max_basis_functions),
        "--max-iterations", str(args.max_iterations),
        "--max-evaluations", str(args.max_evaluations),
        "--parallel-workers", str(args.parallel_workers),
        "--random-seed", str(case.seed),
        "--stress-backend", args.stress_backend,
        "--no-progress",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLCONFIGDIR": "/tmp/pyvale-matplotlib",
            "UV_CACHE_DIR": "/tmp/pyvale-uv-cache",
        }
    )
    started = time.monotonic()
    with log_file.open("w", encoding="utf-8") as stream:
        stream.write(f"command={json.dumps(command)}\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=args.pyvale_root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    runtime = time.monotonic() - started
    status = "complete" if completed.returncode == 0 and result_file.is_file() else "failed"
    return CaseResult(
        case, status, completed.returncode, runtime, str(result_file), str(log_file)
    )


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    dataset: Path,
    cases: list[CampaignCase],
    results: list[CaseResult],
) -> None:
    completed = {item.case.name: item for item in results}
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(),
        "dataset": str(dataset),
        "campaign_name": args.campaign_name,
        "configuration": {
            "gates": list(args.gates),
            "seeds": list(args.seeds),
            "control_gate": args.control_gate,
            "jobs": args.jobs,
            "parallel_workers": args.parallel_workers,
            "max_basis_functions": args.max_basis_functions,
            "max_iterations": args.max_iterations,
            "max_evaluations": args.max_evaluations,
            "stress_backend": args.stress_backend,
        },
        "cases": [
            {
                **asdict(case),
                **(
                    {
                        key: value
                        for key, value in asdict(completed[case.name]).items()
                        if key != "case"
                    }
                    if case.name in completed
                    else {"status": "pending"}
                ),
            }
            for case in cases
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pyvale-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--campaign-name", default="gate_objective_campaign_20260828"
    )
    parser.add_argument("--gates", type=_parse_floats, default=(0.0, 0.005, 0.01, 0.05))
    parser.add_argument("--seeds", type=_parse_ints, default=tuple(range(8)))
    parser.add_argument("--control-gate", type=float, default=0.01)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--max-basis-functions", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--max-evaluations", type=int, default=15_500)
    parser.add_argument("--stress-backend", choices=("numpy", "cython"), default="cython")
    args = parser.parse_args()
    if args.jobs < 1 or args.parallel_workers < 1:
        parser.error("--jobs and --parallel-workers must be positive")
    if not args.gates or any(gate < 0.0 or gate >= 1.0 for gate in args.gates):
        parser.error("--gates must contain values in [0, 1)")
    if not 0.0 <= args.control_gate < 1.0:
        parser.error("--control-gate must lie in [0, 1)")
    args.pyvale_root = args.pyvale_root.expanduser().resolve()
    return args


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def _parse_ints(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lower, upper = (int(part) for part in item.split("-", maxsplit=1))
            values.extend(range(lower, upper + 1))
        else:
            values.append(int(item))
    return tuple(dict.fromkeys(values))


if __name__ == "__main__":
    main()
