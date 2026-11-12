"""Repeated plate validation timings; run from the repository root."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
import platform
import statistics
import subprocess
from time import perf_counter

# Set native thread limits before NumPy/SciPy imports and spawned workers.
NATIVE_THREADS = 1
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = str(NATIVE_THREADS)

import numpy as np
import scipy

import pyvale.valid as val
import pyvale.valid.cython.mavm as mavm


CAMERA_WIDTH = 4
CAMERA_HEIGHT = 4
STRAIN_GRID = (4, 3)
THERMOCOUPLE_GRID = (5, 2)
SIM_EPISTEMIC_SAMPLES = 4
SIM_ALEATORY_SAMPLES = 64
EXP_ALEATORY_SAMPLES = 8
TIME_STEPS = 2
WORKERS = 4
CHUNK_SIZE = 128
TIMING_RUNS = 3
RANDOM_SEED = 2871
X_LIMITS_MM = (10.0, 90.0)
Y_LIMITS_MM = (10.0, 40.0)
STRAIN_NOISE_STD = 2.0e-6
TEMPERATURE_NOISE_STD = 0.3
DISPLACEMENT_NOISE_STD_MM = 0.001
EXP_OFFSET_NOISE_MULTIPLIER = 2.0
MAVM_ALPHA = 0.05
REPORT_PATH = Path.home() / "lfdev/ai-coding/pyvale_valid_perf_run.md"


@dataclass(slots=True)
class BenchmarkCase:
    name: str
    distribution: val.ValCase
    paired: val.ValCase


def create_grid(width: int, height: int) -> np.ndarray:
    horizontal, vertical = np.meshgrid(
        np.linspace(*X_LIMITS_MM, width),
        np.linspace(*Y_LIMITS_MM, height),
        indexing="xy",
    )
    return np.column_stack(
        (horizontal.ravel(), vertical.ravel(), np.zeros(horizontal.size))
    )


def prepare_cases(
    *, camera_width: int, camera_height: int, time_steps: int
) -> tuple[list[BenchmarkCase], np.ndarray]:
    # Spawned metric workers do not need sensor generation or VTK imports.
    import pyvale.data as dataset
    import pyvale.mooseherder as mh
    import pyvale.sensorsim as sens

    nominal_path, biased_path = dataset.thermomechanical_2d_experiment_paths()
    nominal = mh.ExodusLoader(nominal_path).load_all_sim_data()
    biased = mh.ExodusLoader(biased_path).load_all_sim_data()
    for simulation in (nominal, biased):
        sens.scale_length_units(1000.0, simulation, disp_keys=("disp_x", "disp_y"))
    if not 1 <= time_steps <= len(nominal.time):
        raise ValueError("TIME_STEPS must select between 1 and all plate time steps.")
    if not np.array_equal(nominal.time, biased.time):
        raise ValueError("Nominal and biased plate time steps must match.")
    time_indices = np.rint(np.linspace(0, len(nominal.time) - 1, time_steps)).astype(
        np.intp
    )
    times = nominal.time[time_indices].copy()
    specifications = (
        ("strain_gauges", STRAIN_GRID, ("strain_xx",), "1", STRAIN_NOISE_STD),
        (
            "thermocouples", THERMOCOUPLE_GRID, ("temperature",), "degC",
            TEMPERATURE_NOISE_STD,
        ),
        (
            "dic", (camera_width, camera_height), ("disp_x", "disp_y"), "mm",
            DISPLACEMENT_NOISE_STD_MM,
        ),
        (
            "ir", (camera_width, camera_height), ("temperature",), "degC",
            TEMPERATURE_NOISE_STD,
        ),
    )
    simulation_axes = (
        val.EValAxis.POINT, val.EValAxis.COMPONENT, val.EValAxis.TIME,
        val.EValAxis.EPISTEMIC_TRANS, val.EValAxis.ALEATORY,
    )
    experiment_axes = simulation_axes[:3] + (val.EValAxis.ALEATORY,)
    generator = np.random.default_rng(RANDOM_SEED)
    cases = []
    for name, grid, components, units, noise_std in specifications:
        coords = create_grid(*grid)
        observations = []
        for simulation in (nominal, biased):
            if len(components) == 1:
                field = sens.FieldScalar(simulation, components[0], sens.EDim.TWOD)
            else:
                field = sens.FieldVector(simulation, components, sens.EDim.TWOD)
            observations.append(field.sample_field(coords, times))
        nominal_observations, biased_observations = observations
        physical_shape = (len(coords), len(components), len(times))
        if any(observation.shape != physical_shape for observation in observations):
            raise ValueError(f"Unexpected sensor observation shape for {name}.")
        # Synthetic transferable scenarios interpolate nominal/biased responses.
        # These are response-space surrogates rather than separate FE solutions.
        simulation_samples = np.empty(
            (*physical_shape, SIM_EPISTEMIC_SAMPLES, SIM_ALEATORY_SAMPLES)
        )
        for scenario, weight in enumerate(np.linspace(0, 1, SIM_EPISTEMIC_SAMPLES)):
            scenario_mean = (
                (1.0 - weight) * nominal_observations + weight * biased_observations
            )
            simulation_samples[..., scenario, :] = generator.normal(
                0.0, noise_std, (*physical_shape, SIM_ALEATORY_SAMPLES)
            )
            simulation_samples[..., scenario, :] += scenario_mean[..., None]
        experiment_samples = generator.normal(
            0.0, noise_std, (*physical_shape, EXP_ALEATORY_SAMPLES)
        )
        experiment_samples += biased_observations[..., None]
        experiment = val.ValArray(
            experiment_samples, experiment_axes,
            val.EpistemicOffsets(
                noise_std * EXP_OFFSET_NOISE_MULTIPLIER,
                noise_std * EXP_OFFSET_NOISE_MULTIPLIER,
            ),
        )
        domain = val.ValDomain({name: val.ValObservable(
            val.ValGeometry(
                val.EValGeometry.POINT, coords, coord_units="mm", coord_frame="plate"
            ),
            val.EValOperator.SAMPLE, components, times, units, "s",
        )})
        distribution = val.ValCase(
            domain, {name: val.ValArray(simulation_samples, simulation_axes)},
            {name: experiment},
        )
        # Explicit first-N pairing, retaining all epistemic/offset scenarios.
        paired = val.ValCase(
            domain,
            {name: val.ValArray(
                simulation_samples[..., :EXP_ALEATORY_SAMPLES], simulation_axes
            )},
            {name: experiment},
        )
        cases.append(BenchmarkCase(name, distribution, paired))
    return cases, times


def check_outputs(result: val.ValResult) -> tuple[float, ...]:
    checksums = []
    for fields in result.fields.values():
        for field in fields.values():
            for array in (field.metric, field.d_plus, field.d_minus):
                if array is not None:
                    if not np.all(np.isfinite(array)):
                        raise AssertionError("Benchmark produced nonfinite metrics.")
                    checksums.append(float(np.sum(array)))
    return tuple(checksums)


def describe_statistics(durations: list[float]) -> str:
    mean = statistics.mean(durations)
    lines = [
        f"runs={len(durations)}, mean={mean:.3f} s, "
        f"median={statistics.median(durations):.3f} s",
        f"min={min(durations):.3f} s, max={max(durations):.3f} s, "
        f"range={max(durations) - min(durations):.3f} s",
    ]
    if len(durations) > 1:
        deviation = statistics.stdev(durations)
        lines.append(
            f"sample variance={statistics.variance(durations):.6f} s^2, "
            f"sample stddev={deviation:.3f} s, CV={100 * deviation / mean:.2f}%"
        )
    else:
        lines.append("Run-to-run variance unavailable with one run.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=TIMING_RUNS)
    parser.add_argument("--camera-width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--time-steps", type=int, default=TIME_STEPS)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    arguments = parser.parse_args()
    counts = (
        arguments.runs, arguments.camera_width, arguments.camera_height,
        arguments.time_steps, SIM_EPISTEMIC_SAMPLES, SIM_ALEATORY_SAMPLES,
        EXP_ALEATORY_SAMPLES, WORKERS, CHUNK_SIZE, *STRAIN_GRID, *THERMOCOUPLE_GRID,
    )
    if any(count < 1 for count in counts):
        parser.error("All run, grid, sample, worker and chunk counts must be positive.")
    if EXP_ALEATORY_SAMPLES > SIM_ALEATORY_SAMPLES:
        parser.error("Paired comparison requires experiment count <= simulation count.")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
        cwd=Path(__file__).resolve().parents[1],
    ).strip()
    processor = platform.processor()
    cpu_info_path = Path("/proc/cpuinfo")
    if cpu_info_path.is_file():
        for line in cpu_info_path.read_text().splitlines():
            if line.startswith("model name"):
                processor = line.partition(":")[2].strip()
                break
    native_fingerprint = sha256(Path(mavm.__file__).read_bytes()).hexdigest()
    metadata = (
        f"UTC: {datetime.now(timezone.utc).isoformat()}\n"
        f"Revision: {revision}\n"
        f"Platform: {platform.platform()}; CPU: {processor}\n"
        f"Python: {platform.python_version()}; NumPy: {np.__version__}; "
        f"SciPy: {scipy.__version__}\nMAVM module: {mavm.__file__}\n"
        f"MAVM binary SHA256: {native_fingerprint}\n"
        f"Camera grids: {arguments.camera_width} x {arguments.camera_height}; "
        f"strain grid: {STRAIN_GRID}; thermocouple grid: {THERMOCOUPLE_GRID}\n"
        f"Simulation epistemic/aleatory: {SIM_EPISTEMIC_SAMPLES}/"
        f"{SIM_ALEATORY_SAMPLES}; experiment aleatory: {EXP_ALEATORY_SAMPLES}\n"
        f"Workers: {WORKERS}; chunk size: {CHUNK_SIZE}; seed: {RANDOM_SEED}; "
        f"native threads per worker: {NATIVE_THREADS}"
    )
    print(metadata, flush=True)
    preparation_start = perf_counter()
    cases, times = prepare_cases(
        camera_width=arguments.camera_width,
        camera_height=arguments.camera_height,
        time_steps=arguments.time_steps,
    )
    preparation_seconds = perf_counter() - preparation_start
    input_bytes = sum(
        case.distribution.sim[case.name].data.nbytes
        + case.distribution.exp[case.name].data.nbytes for case in cases
    )
    setup = (
        f"Selected simulation times (s): {times.tolist()}\n"
        f"Input arrays: {input_bytes / 1e6:.3f} MB; "
        f"preparation: {preparation_seconds:.3f} s"
    )
    print(setup, flush=True)
    report_path = arguments.report.expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Validation performance baseline (in progress)\n\n"
        + metadata + "\n\n" + setup + "\n\n"
        + f"Requested timing runs: {arguments.runs}.\n"
        + "Timing run in progress; optimisation decisions are recorded separately.\n",
        encoding="utf-8",
    )
    distribution_engine = val.ValidationEngine(
        [val.MetricMAVM(alpha=MAVM_ALPHA), val.MetricAVM(),
         val.MetricKS(), val.MetricCVM()],
        opts=val.ValEngineOpts(workers=WORKERS, chunk_size=CHUNK_SIZE),
    )
    # A fixed positive scale avoids singular relative error at initial zero fields.
    paired_engines = {
        case.name: val.ValidationEngine(
            [val.MetricRMSE(), val.MetricRelErr(
                normalization=val.ERelativeErrorNormalization.CONSTANT,
                constant={
                    "strain_gauges": STRAIN_NOISE_STD,
                    "thermocouples": TEMPERATURE_NOISE_STD,
                    "dic": DISPLACEMENT_NOISE_STD_MM,
                    "ir": TEMPERATURE_NOISE_STD,
                }[case.name],
            )],
            opts=val.ValEngineOpts(workers=WORKERS, chunk_size=CHUNK_SIZE),
        ) for case in cases
    }
    distribution_case = val.ValCase(
        val.ValDomain({
            case.name: case.distribution.domain.observables[case.name]
            for case in cases
        }),
        {case.name: case.distribution.sim[case.name] for case in cases},
        {case.name: case.distribution.exp[case.name] for case in cases},
    )
    # One distribution engine call shares worker startup across all observables.
    # Paired calls retain each observable's physical normalization constant.
    workloads = [("all", "distribution", distribution_engine, distribution_case)]
    workloads.extend(
        (case.name, "paired", paired_engines[case.name], case.paired)
        for case in cases
    )
    durations: list[float] = []
    reference_checksums: dict[tuple[str, str], tuple[float, ...]] = {}
    records = []
    for run in range(arguments.runs):
        run_start = perf_counter()
        elapsed = 0.0
        for name, group, engine, val_case in workloads:
            print(f"Run {run + 1}: {name}/{group} starting", flush=True)
            start = perf_counter()
            result = engine.run(val_case)
            seconds = perf_counter() - start
            elapsed += seconds
            checksums = check_outputs(result)
            key = (name, group)
            if run == 0:
                reference_checksums[key] = checksums
            elif checksums != reference_checksums[key]:
                raise AssertionError("Metric checksums changed between runs.")
            del result
            record = f"Run {run + 1}: {name}/{group}: {seconds:.3f} s"
            records.append(record)
            print(record, flush=True)
            with report_path.open("a", encoding="utf-8") as report_file:
                report_file.write("\n" + record + "\n")
        wall_seconds = perf_counter() - run_start
        durations.append(wall_seconds)
        print(f"Run {run + 1} total engine time: {elapsed:.3f} s", flush=True)
        print(f"Run {run + 1} wall time: {wall_seconds:.3f} s", flush=True)
        records.append(
            f"Run {run + 1} total engine time: {elapsed:.3f} s; "
            f"wall time: {wall_seconds:.3f} s"
        )
        summary = describe_statistics(durations)
        report = (
            "# Validation performance baseline\n\n" + metadata + "\n\n" + setup
            + "\n\nAll six engine metrics are included. Distribution metrics use "
            "all simulation/experiment samples; RMSE and relative error pair the "
            "first experiment-count simulation samples with experiment samples. "
            "All groups retain the simulation epistemic axis and nominal/lower/"
            "upper experimental offsets. Strain gauges measure strain_xx; DIC "
            "measures disp_x/disp_y; thermocouples and IR measure temperature. "
            "Synthetic epistemic responses interpolate the nominal and biased "
            "plate solutions. Relative error is normalized by measurement noise."
            "\n\nDistribution metrics share one engine call across observables; "
            "paired metrics execute separately for each physical normalization. "
            "Timings include engine preparation, shared-memory copies, worker "
            "spawn and output allocation. Run totals are wall times including "
            "output checks; sensor/input preparation is reported separately. "
            "Inputs are reused unchanged "
            "across runs; no warmup runs are discarded.\n\n"
            + "\n".join(records) + "\n\nRun totals (s): "
            + ", ".join(f"{duration:.6f}" for duration in durations)
            + "\n\n" + summary
            + "\n\nThis file records timings; retention decisions are in "
            "pyvale_valid_perf_report.md.\n"
        )
        report_path.write_text(report, encoding="utf-8")
        print(summary, flush=True)
    print(f"Report saved to {report_path}", flush=True)


if __name__ == "__main__":
    main()
