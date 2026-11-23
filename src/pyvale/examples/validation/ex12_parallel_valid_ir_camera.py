# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Parallel IR camera validation
================================================================================
"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

import pyvale.validation as vl

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_COUNT = 20
SIM_EMISSIVITY_COUNT = 12
EXP_EMISSIVITY_COUNT = 12
NOISE_STD_DEGC = 0.05
PLATE_WIDTH_M = 0.64
PLATE_HEIGHT_M = 0.48


def run_ir_camera_validation() -> None:
    setup_start = perf_counter()

    normalized_x, normalized_y = np.meshgrid(
        np.linspace(0.0, 1.0, CAMERA_WIDTH),
        np.linspace(0.0, 1.0, CAMERA_HEIGHT),
    )
    ground_truth = (
        20.0
        + 35.0 * normalized_x
        + 30.0 * normalized_y
        + 15.0 * normalized_x * normalized_y
    )
    coords = np.column_stack(
        (
            PLATE_WIDTH_M * normalized_x.ravel(),
            PLATE_HEIGHT_M * normalized_y.ravel(),
            np.zeros(ground_truth.size),
        )
    )

    kelvin_offset = 273.15
    reflected_kelvin = 20.0 + kelvin_offset
    true_emissivity = 0.95
    radiance_contrast = (
        true_emissivity * ((ground_truth + kelvin_offset) ** 4 - reflected_kelvin**4)
    ).reshape(-1, 1)
    sim_assumed_emissivities = np.r_[
        true_emissivity, np.linspace(0.86, 0.94, SIM_EMISSIVITY_COUNT - 1)
    ]
    exp_assumed_emissivities = np.r_[
        true_emissivity, np.linspace(0.89, 0.94, EXP_EMISSIVITY_COUNT - 1)
    ]
    sim_epistemic_temperature_shift = (
        (reflected_kelvin**4 + radiance_contrast / sim_assumed_emissivities) ** 0.25
        - kelvin_offset
        - ground_truth.reshape(-1, 1)
    ).astype(np.float32)
    exp_epistemic_temperature_shift = (
        (reflected_kelvin**4 + radiance_contrast / exp_assumed_emissivities) ** 0.25
        - kelvin_offset
        - ground_truth.reshape(-1, 1)
    ).astype(np.float32)

    generator = np.random.default_rng(2871)
    frame_sample_shape = (ground_truth.size, FRAME_COUNT)
    ground_truth_by_pixel = ground_truth.astype(np.float32).reshape(-1, 1)
    sim_aleatory_noise = NOISE_STD_DEGC * generator.standard_normal(
        frame_sample_shape, dtype=np.float32
    )
    exp_aleatory_noise = NOISE_STD_DEGC * generator.standard_normal(
        frame_sample_shape, dtype=np.float32
    )
    sim_nominal_temperature_samples = ground_truth_by_pixel + sim_aleatory_noise
    exp_nominal_temperature_samples = ground_truth_by_pixel + exp_aleatory_noise
    sim_temperature_samples = (
        sim_nominal_temperature_samples[:, None, :]
        + sim_epistemic_temperature_shift[:, :, None]
    )
    exp_temperature_samples = (
        exp_nominal_temperature_samples[:, None, :]
        + exp_epistemic_temperature_shift[:, :, None]
    )
    physical_axes = (vl.EValidAxis.POINT, vl.EValidAxis.COMPONENT, vl.EValidAxis.TIME)
    observable_tag = "plate_temperature"
    valid_case = vl.ValidCase(
        {
            observable_tag: vl.ValidObservable(
                components=("temperature",),
                geometry=vl.EValidGeometry.SURFACE,
                coords=coords,
                times=np.array([0.0]),
                phys_units="degC",
                coord_units="m",
                time_units="s",
            )
        },
        {
            observable_tag: vl.ValidArray(
                sim_temperature_samples[:, None, None, :, :],
                physical_axes + (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY),
            )
        },
        {
            observable_tag: vl.ValidArray(
                exp_temperature_samples[:, None, None, :, :],
                physical_axes + (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY),
            )
        },
    )
    setup_seconds = perf_counter() - setup_start
    print(
        f"Setup ({CAMERA_WIDTH} x {CAMERA_HEIGHT}, {FRAME_COUNT} frames per source): "
        f"{setup_seconds:.3f} s | emissivity cases: "
        f"{SIM_EMISSIVITY_COUNT} sim × {EXP_EMISSIVITY_COUNT} exp",
        flush=True,
    )

    print("Calculating validation metrics serial...", flush=True)
    serial_start = perf_counter()
    serial_result = vl.ValidEngine(
        [vl.MetricAVM()], opts=vl.ValidEngineOpts(workers=1)
    ).run(valid_case)
    serial_seconds = perf_counter() - serial_start
    print(f"Serial AVM: {serial_seconds:.3f} s", flush=True)

    print("Calculating validation metrics in parallel (4 workers)...", flush=True)
    parallel_start = perf_counter()
    parallel_result = vl.ValidEngine(
        [vl.MetricAVM()], opts=vl.ValidEngineOpts(workers=4)
    ).run(valid_case)
    parallel_seconds = perf_counter() - parallel_start
    print(f"Parallel AVM (4 workers): {parallel_seconds:.3f} s")
    print(f"Combined solve: {serial_seconds + parallel_seconds:.3f} s")
    print(f"Four-worker speedup: {serial_seconds / parallel_seconds:.2f}×")

    serial_avm = serial_result.metrics[observable_tag][vl.EMetricType.AVM].outputs[
        vl.EMetricOut.DISTANCE
    ]
    parallel_avm = parallel_result.metrics[observable_tag][vl.EMetricType.AVM].outputs[
        vl.EMetricOut.DISTANCE
    ]
    np.testing.assert_array_equal(serial_avm, parallel_avm)

    output_path = Path.cwd() / "pyvale-output" / "valid_ex12_parallel_ir_camera"
    output_path.mkdir(parents=True, exist_ok=True)
    plot_opts = vl.ValidPlotOpts()
    figure, axes = plt.subplots(
        2, 2, figsize=(7.25, 5.0), layout="constrained", dpi=plot_opts.resolution
    )
    avm_by_pixel = serial_avm[:, 0, 0]
    nominal_avm = avm_by_pixel[:, 0, 0].reshape(CAMERA_HEIGHT, CAMERA_WIDTH)
    worst_case_avm = avm_by_pixel.max(axis=(1, 2)).reshape(CAMERA_HEIGHT, CAMERA_WIDTH)
    for panel, image, title in zip(
        axes.flat,
        (
            ground_truth,
            exp_epistemic_temperature_shift[:, 1].reshape(CAMERA_HEIGHT, CAMERA_WIDTH),
            nominal_avm,
            worst_case_avm,
        ),
        (
            "Ground truth",
            "Largest experimental emissivity shift",
            "AVM: nominal",
            "AVM: worst emissivity pair",
        ),
        strict=True,
    ):
        heatmap = panel.imshow(
            image, origin="lower", extent=(0.0, PLATE_WIDTH_M, 0.0, PLATE_HEIGHT_M)
        )
        panel.set(title=title, xlabel="x [m]", ylabel="y [m]")
        figure.colorbar(heatmap, ax=panel, label="Temperature [degC]")
    figure.savefig(output_path / "ir_camera_avm.png")
    plt.close(figure)


if __name__ == "__main__":
    run_ir_camera_validation()
