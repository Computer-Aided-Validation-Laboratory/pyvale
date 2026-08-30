"""Run clean/noisy exploratory objective variants for notched-EBW overnight."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_DATASET = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/"
    "synthetic-fe/wdbn1-idealised-yield/pyvale-vfm"
)


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    label: str
    windows: str
    weights: str
    force_weight: float
    sensitivity_weighting: bool = False


@dataclass(frozen=True, slots=True)
class CampaignCase:
    name: str
    policy: str
    gate: float
    seed: int
    objective: str
    condition: str
    noise_scale: float
    windows: str
    weights: str
    force_weight: float
    sensitivity_weighting: bool


@dataclass(slots=True)
class CaseResult:
    case: CampaignCase
    status: str
    return_code: int
    runtime_seconds: float
    result_file: str
    log_file: str


OBJECTIVES = (
    ObjectiveSpec("current", "29,57", "29,57", 0.10),
    ObjectiveSpec("multiscale_length", "15,29,57", "15,29,57", 0.10),
    ObjectiveSpec("multiscale_equal", "15,29,57", "1,1,1", 0.10),
    ObjectiveSpec("fine_emphasis", "15,29,57", "5,3,2", 0.10),
    ObjectiveSpec("broad_fre_guard", "15,29,57", "1,2,7", 0.25),
    ObjectiveSpec("sensitivity_equal", "15,29,57", "1,1,1", 0.10, True),
)


def main() -> None:
    args = _parse_args()
    dataset = args.dataset.expanduser().resolve()
    if not (dataset / "prepared/experiment_data.yaml").is_file():
        raise FileNotFoundError(f"Prepared dataset not found below {dataset}.")
    campaign_root = dataset / "identification/prepared" / args.campaign_name
    logs = campaign_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cases = _build_cases(args.noise_scale, args.seed)
    manifest = campaign_root / "campaign_manifest.json"
    _write_manifest(manifest, args, dataset, cases, [])
    print(
        f"Starting {len(cases)} cases: {len(OBJECTIVES)} objectives × clean/noisy; "
        f"{args.jobs} processes × {args.parallel_workers} workers.",
        flush=True,
    )
    started = time.monotonic()
    results: list[CaseResult] = []
    last_progress = started
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        pending = {
            executor.submit(_run_case, case, args, dataset, logs): case
            for case in cases
        }
        while pending:
            done, pending = wait(
                pending,
                timeout=args.progress_interval_seconds,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                result = future.result()
                results.append(result)
                print(
                    f"{result.status:8s} {result.case.name} "
                    f"({result.runtime_seconds / 60:.1f} min)",
                    flush=True,
                )
                _write_manifest(manifest, args, dataset, cases, results)
            now = time.monotonic()
            if pending and now - last_progress >= args.progress_interval_seconds:
                complete = sum(item.status in {"complete", "skipped"} for item in results)
                failed = sum(item.status == "failed" for item in results)
                active = sum(future.running() for future in pending)
                print(
                    f"{datetime.now():%F %T} progress elapsed={(now-started)/60:.1f}min "
                    f"complete={complete}/{len(cases)} failed={failed} "
                    f"active={active} queued={len(pending)-active}",
                    flush=True,
                )
                last_progress = now
    failures = [item for item in results if item.return_code]
    print(f"Manifest: {manifest}", flush=True)
    if failures:
        for item in failures:
            print(f"FAILED {item.case.name}: {item.log_file}", file=sys.stderr)
        raise SystemExit(1)


def _build_cases(noise_scale: float, seed: int) -> list[CampaignCase]:
    cases = []
    for spec in OBJECTIVES:
        for condition, scale in (("clean", 0.0), ("noise", noise_scale)):
            cases.append(
                CampaignCase(
                    name=f"obj_{spec.label}_{condition}_seed{seed:02d}",
                    policy=f"{spec.label}_{condition}",
                    gate=0.0,
                    seed=seed,
                    objective=spec.label,
                    condition=condition,
                    noise_scale=scale,
                    windows=spec.windows,
                    weights=spec.weights,
                    force_weight=spec.force_weight,
                    sensitivity_weighting=spec.sensitivity_weighting,
                )
            )
    return cases


def _run_case(case, args, dataset, logs) -> CaseResult:
    run_name = f"prepared/{args.campaign_name}/{case.name}"
    result_file = dataset / "identification" / run_name / "identification_result.yaml"
    log_file = logs / f"{case.name}.log"
    if result_file.is_file():
        return CaseResult(case, "skipped", 0, 0.0, str(result_file), str(log_file))
    command = [
        "uv", "run", "--no-sync", "python",
        "dev/vfm/call_notched_ebw_bivariate_identification.py",
        "--input", str(dataset / "prepared"),
        "--output-root", str(dataset / "identification"),
        "--run-name", run_name,
        "--kernel-type", "bivariate_spd",
        "--basis-growth-policy", "sensitivity_correction",
        "--minimum-objective-improvement", "0",
        "--egi-windows", case.windows,
        "--egi-window-weights", case.weights,
        "--force-weight", str(case.force_weight),
        "--max-basis-functions", str(args.max_basis_functions),
        "--max-iterations", str(args.max_iterations),
        "--max-evaluations", str(args.max_evaluations),
        "--parallel-workers", str(args.parallel_workers),
        "--random-seed", str(case.seed),
        "--stress-backend", args.stress_backend,
        "--artificial-noise-seed", str(args.noise_seed),
        "--no-progress",
    ]
    if case.noise_scale > 0:
        command.extend([
            "--artificial-noise-model", str(args.noise_model),
            "--artificial-noise-scale", str(case.noise_scale),
        ])
    if case.sensitivity_weighting:
        command.append("--sensitivity-spatial-weighting")
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": "/tmp/pyvale-matplotlib",
        "UV_CACHE_DIR": "/tmp/pyvale-uv-cache",
    })
    started = time.monotonic()
    with log_file.open("w", encoding="utf-8") as stream:
        stream.write(f"command={json.dumps(command)}\n")
        stream.flush()
        completed = subprocess.run(
            command, cwd=args.pyvale_root, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    runtime = time.monotonic() - started
    status = "complete" if completed.returncode == 0 and result_file.is_file() else "failed"
    return CaseResult(case, status, completed.returncode, runtime, str(result_file), str(log_file))


def _write_manifest(path, args, dataset, cases, results) -> None:
    completed = {item.case.name: item for item in results}
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(),
        "dataset": str(dataset),
        "campaign_name": args.campaign_name,
        "purpose": "Exploratory objective aggregation under clean and WDBN1-like noisy synthetic data",
        "configuration": {
            "jobs": args.jobs,
            "parallel_workers": args.parallel_workers,
            "max_basis_functions": args.max_basis_functions,
            "max_iterations": args.max_iterations,
            "max_evaluations": args.max_evaluations,
            "noise_model": str(args.noise_model),
            "noise_scale": args.noise_scale,
            "noise_seed": args.noise_seed,
            "optimiser_seed": args.seed,
            "objectives": [asdict(item) for item in OBJECTIVES],
        },
        "cases": [
            {
                **asdict(case),
                **(
                    {key: value for key, value in asdict(completed[case.name]).items() if key != "case"}
                    if case.name in completed else {"status": "pending"}
                ),
            }
            for case in cases
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pyvale-root", type=Path, default=Path.cwd())
    parser.add_argument("--campaign-name", default="local_objective_noise_20260829")
    parser.add_argument("--noise-model", type=Path, default=Path("dev/vfm/data/wdbn1_noise_model_20260828.yaml"))
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--noise-seed", type=int, default=20260828)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--max-basis-functions", type=int, default=7)
    parser.add_argument("--max-iterations", type=int, default=180)
    parser.add_argument("--max-evaluations", type=int, default=16_100)
    parser.add_argument("--stress-backend", choices=("numpy", "cython"), default="cython")
    parser.add_argument("--progress-interval-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if min(args.jobs, args.parallel_workers, args.max_basis_functions, args.max_iterations, args.max_evaluations) < 1:
        parser.error("Jobs, workers, basis count, iterations and evaluations must be positive.")
    if args.noise_scale <= 0:
        parser.error("--noise-scale must be positive; clean cases are included automatically.")
    args.pyvale_root = args.pyvale_root.expanduser().resolve()
    args.noise_model = (args.pyvale_root / args.noise_model).resolve() if not args.noise_model.is_absolute() else args.noise_model.resolve()
    return args


if __name__ == "__main__":
    main()
