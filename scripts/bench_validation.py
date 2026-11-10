"""Optional DIC-like validation benchmark using C-order memory-mapped arrays.

Run ``.venv/bin/python scripts/bench_validation.py --grid 128 --samples 16``.
The benchmark reports elapsed time and the size of input/output arrays. It is
excluded from CI and is not a numerical correctness test.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

import pyvale.valid as val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    if args.grid <= 0 or args.samples <= 0:
        raise ValueError("Grid size and sample count must be positive.")

    points = args.grid * args.grid
    axis = np.linspace(0.0, 1.0, args.grid)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    coords = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(points)))
    domain = val.ValDomain(
        {
            "strain": val.ValObservable(
                val.ValGeometry(val.EValGeometry.POINT, coords),
                val.EValOperator.SAMPLE,
                ("strain_xx",),
                None,
                "1",
            )
        }
    )
    shape = (points, 1, args.samples)
    axes = (val.EValAxis.POINT, val.EValAxis.COMPONENT, val.EValAxis.ALEATORY)
    with TemporaryDirectory() as temporary:
        sim_file = np.memmap(
            Path(temporary) / "sim.dat", dtype="float64", mode="w+", shape=shape
        )
        exp_file = np.memmap(
            Path(temporary) / "exp.dat", dtype="float64", mode="w+", shape=shape
        )
        rng = np.random.default_rng(2751)
        for point in range(points):
            samples = rng.normal(size=args.samples)
            sim_file[point, 0] = samples
            exp_file[point, 0] = samples + 0.1
        val_case = val.ValCase(
            domain,
            val.val_from_arrays({"strain": val.ValArray(sim_file, axes)}),
            val.val_from_arrays({"strain": val.ValArray(exp_file, axes)}),
        )
        start = perf_counter()
        field = (
            val.ValidationEngine([val.MetricAVM()])
            .run(val_case)
            .get("strain", val.EMetricType.AVM)
        )
        elapsed = perf_counter() - start
        if not np.allclose(field.metric, 0.1, atol=1.0e-12):
            raise AssertionError("AVM shift sanity check failed.")
        print(f"points={points}, samples={args.samples}, seconds={elapsed:.3f}")
        print(f"input_bytes={sim_file.nbytes + exp_file.nbytes}")
        print(f"output_bytes={field.metric.nbytes}")


if __name__ == "__main__":
    main()
