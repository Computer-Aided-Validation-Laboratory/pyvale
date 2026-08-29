"""Build a multi-start SPD Gaussian direct-map-fit reference for notched EBW."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scipy.optimize import Bounds, minimize

from pyvale.vfm import ExperimentData, SpatialParameterisationBasisFunction
from pyvale.vfm.campaignprogress import ProgressEstimate, atomic_write_json


@dataclass(slots=True)
class FitRow:
    basis_count: int
    start_index: int
    success: bool
    evaluations: int
    runtime_seconds: float
    homogeneous_yield_mpa: float
    roi_rmse_mpa: float
    roi_mape_percent: float
    yielded_rmse_mpa: float
    high_plastic_rmse_mpa: float


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_dir = args.input.expanduser().resolve()
    experiment = ExperimentData.load_from_file(input_dir / "experiment_data.yaml")
    known = dict(np.load(input_dir / "known_parameter_maps.npz"))
    truth = np.asarray(known["yield_strength"], dtype=np.float64)
    x = np.asarray(experiment.specimen_geometry.x, dtype=np.float64)
    y = np.asarray(experiment.specimen_geometry.y, dtype=np.float64)
    roi = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(x, y)
    yielded = roi & (truth < np.nanmax(truth[roi]) - 1.0)
    high_plastic = roi & (truth <= np.nanpercentile(truth[roi], 25.0))
    rows_path = output / "direct_fit_starts.csv"
    existing = _read_rows(rows_path) if args.resume else []
    completed = {(row.basis_count, row.start_index) for row in existing}
    maps: dict[tuple[int, int], np.ndarray] = {}
    npz_path = output / "direct_fit_start_maps.npz"
    if args.resume and npz_path.is_file():
        archive = np.load(npz_path)
        maps.update({
            tuple(int(item) for item in key.removeprefix("bf").split("_start")): np.asarray(archive[key])
            for key in archive.files if key.startswith("bf")
        })

    total = (args.max_bases + 1) * args.starts
    started = time.monotonic()
    rows = list(existing)
    rng = np.random.default_rng(args.random_seed)
    for basis_count in range(args.max_bases + 1):
        seed_basis = _seed_basis(x, y, truth, roi, basis_count, args)
        for start_index in range(args.starts):
            if (basis_count, start_index) in completed:
                continue
            fit_started = time.monotonic()
            basis = copy.deepcopy(seed_basis)
            fitted, homogeneous, success, evaluations = _fit_joint(
                basis, truth, roi, args, rng, start_index
            )
            runtime = time.monotonic() - fit_started
            row = FitRow(
                basis_count=basis_count,
                start_index=start_index,
                success=success,
                evaluations=evaluations,
                runtime_seconds=runtime,
                homogeneous_yield_mpa=homogeneous,
                **_errors(truth, fitted, roi, yielded, high_plastic),
            )
            rows.append(row)
            maps[(basis_count, start_index)] = fitted
            _write_rows(rows_path, rows)
            np.savez_compressed(
                npz_path,
                truth_yield_strength=truth,
                roi_mask=roi,
                yielded_mask=yielded,
                high_plastic_mask=high_plastic,
                **{f"bf{bf}_start{start}": value for (bf, start), value in maps.items()},
            )
            estimate = ProgressEstimate.from_counts(len(rows), total, started)
            print(
                estimate.line(prefix="direct-fit")
                + f" bf={basis_count} start={start_index} "
                + f"rmse={row.roi_rmse_mpa:.3f}MPa success={success}",
                flush=True,
            )
            atomic_write_json(output / "screen_manifest.json", _manifest(args, rows, total))

    best_rows = [
        min((row for row in rows if row.basis_count == count), key=lambda row: row.roi_rmse_mpa)
        for count in range(args.max_bases + 1)
    ]
    _write_rows(output / "direct_fit_reference.csv", best_rows)
    np.savez_compressed(
        output / "direct_fit_reference.npz",
        basis_count=np.asarray([row.basis_count for row in best_rows]),
        fitted_yield_strength=np.stack([
            maps[(row.basis_count, row.start_index)] for row in best_rows
        ]),
        truth_yield_strength=truth,
        roi_mask=roi,
        yielded_mask=yielded,
        high_plastic_mask=high_plastic,
    )
    atomic_write_json(output / "screen_manifest.json", _manifest(args, rows, total, complete=True))
    print(f"direct-fit complete output={output}", flush=True)


def _seed_basis(x, y, truth, roi, count, args):
    basis = SpatialParameterisationBasisFunction(
        x=x, y=y, kernel_type="bivariate_spd",
        centre_bounds_span_factor=args.centre_bounds_span_factor,
    )
    if count:
        basis.fit_to_map(
            truth - args.initial_homogeneous_yield,
            parameter_range=args.yield_upper_bound - args.yield_lower_bound,
            max_basis_functions=count,
            minimum_relative_improvement=0.0,
            fit_mask=roi,
        )
    return basis


def _fit_joint(basis, truth, roi, args, rng, start_index):
    basis_dofs = basis._collect_internal_degrees_of_freedom(True)
    lower = np.asarray([args.yield_lower_bound, *[d.lower_bound for d in basis_dofs]])
    upper = np.asarray([args.yield_upper_bound, *[d.upper_bound for d in basis_dofs]])
    physical = np.asarray([args.initial_homogeneous_yield, *[d.value for d in basis_dofs]])
    initial = (physical - lower) / (upper - lower)
    if start_index:
        initial = np.clip(
            initial + rng.normal(0.0, args.start_perturbation, initial.shape), 0.0, 1.0
        )

    shape = np.asarray(truth.shape, dtype=np.uint32)
    def update(candidate):
        values = lower + np.asarray(candidate) * (upper - lower)
        if basis_dofs:
            basis._update_internal_from_degrees_of_freedom(values[1:], True)
        return float(values[0]) + basis.to_map(shape)

    def cost(candidate):
        error = update(candidate)[roi] - truth[roi]
        return float(np.sqrt(np.mean(error**2)))

    result = minimize(
        cost, initial, method="L-BFGS-B", bounds=Bounds(0.0, 1.0),
        options={"maxiter": args.max_iterations, "ftol": args.ftol},
    )
    fitted = update(result.x)
    homogeneous = float(lower[0] + result.x[0] * (upper[0] - lower[0]))
    return fitted, homogeneous, bool(result.success), int(result.nfev)


def _errors(truth, fitted, roi, yielded, high_plastic):
    def rmse(mask):
        return float(np.sqrt(np.mean((fitted[mask] - truth[mask])**2)))
    return {
        "roi_rmse_mpa": rmse(roi),
        "roi_mape_percent": float(np.mean(np.abs(fitted[roi] - truth[roi]) / np.abs(truth[roi])) * 100.0),
        "yielded_rmse_mpa": rmse(yielded),
        "high_plastic_rmse_mpa": rmse(high_plastic),
    }


def _read_rows(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [FitRow(**{
            field: (
                int(value) if field in {"basis_count", "start_index", "evaluations"}
                else value.lower() == "true" if field == "success" else float(value)
            ) for field, value in row.items()
        }) for row in csv.DictReader(stream)]


def _write_rows(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    temporary.replace(path)


def _manifest(args, rows, total, complete=False):
    return {
        "tool": "build_notched_ebw_direct_fit_reference",
        "status": "complete" if complete else "running",
        "input": str(args.input), "output": str(args.output),
        "configuration": {
            "max_bases": args.max_bases, "starts": args.starts,
            "random_seed": args.random_seed,
            "start_perturbation": args.start_perturbation,
            "kernel_type": "bivariate_spd", "fit_domain": "roi",
            "yield_bounds_mpa": [args.yield_lower_bound, args.yield_upper_bound],
        },
        "progress": {"completed": len(rows), "total": total},
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bases", type=int, default=8)
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260829)
    parser.add_argument("--start-perturbation", type=float, default=0.08)
    parser.add_argument("--initial-homogeneous-yield", type=float, default=543.0)
    parser.add_argument("--yield-lower-bound", type=float, default=200.0)
    parser.add_argument("--yield-upper-bound", type=float, default=2000.0)
    parser.add_argument("--centre-bounds-span-factor", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--ftol", type=float, default=1e-10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_bases < 0 or args.starts < 1 or args.max_iterations < 1:
        parser.error("Basis, start, and iteration counts are invalid.")
    if not 0.0 <= args.start_perturbation <= 0.5:
        parser.error("--start-perturbation must lie in [0, 0.5].")
    return args


if __name__ == "__main__":
    main()
