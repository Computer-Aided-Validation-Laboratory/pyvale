# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Parallel IR camera validation
================================================================================

Compare serial and four-worker AVM calculations for an infrared camera viewing
a heated plate.
"""

from itertools import count
from pathlib import Path
from time import perf_counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.validation as vl

# %%
matplotlib.use("Agg")

SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex12_parallel_ir_camera"
output_path.mkdir(parents=True, exist_ok=True)
fig_nums = count(1)

# %%
# 1. Configure the camera and sampling study
# ------------------------------------------
#
# Each pixel represents one validation point. The steady-state experiment uses
# repeated frames for aleatory noise and different assumed emissivities for
# epistemic uncertainty.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_COUNT = 20
SIM_EMISSIVITY_COUNT = 12
EXP_EMISSIVITY_COUNT = 12
NOISE_STD_DEGC = 0.05
PLATE_WIDTH_M = 0.64
PLATE_HEIGHT_M = 0.48


# %%
# 2. Generate plate temperatures and camera samples
# -------------------------------------------------
#
# The bilinear temperature field ranges from 20 degC at the bottom left to 100
# degC at the top right. The radiance inversion yields a temperature shift for
# each pixel and emissivity assumption. Broadcast those shifts across the
# noisy frames to produce arrays ordered by pixel, emissivity case, and frame.
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

sim_assumed_emissivities = np.concatenate(
    ([true_emissivity], np.linspace(0.86, 0.94, SIM_EMISSIVITY_COUNT - 1))
)
exp_assumed_emissivities = np.concatenate(
    ([true_emissivity], np.linspace(0.89, 0.94, EXP_EMISSIVITY_COUNT - 1))
)

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

# %%
# 3. Name the validation dimensions
# ---------------------------------
#
# Insert singleton component and time axes for one temperature component at
# one steady time. Each source then has point, component, time, transferable
# case, and aleatory axes. Pixel coordinates retain the spatial layout.
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

print(80*"-")
print(
    f"Setup ({CAMERA_WIDTH} x {CAMERA_HEIGHT}, {FRAME_COUNT} frames per source): "
    f"{setup_seconds:.3f} s | emissivity cases: "
    f"{SIM_EMISSIVITY_COUNT} sim × {EXP_EMISSIVITY_COUNT} exp",
    flush=True,
)
print(80*"-")

# %%
# 4. Time serial and parallel validation
# --------------------------------------
#
# Run the same case with one and four workers. This equal-sample AVM workload
# uses batched NumPy calculations and shares observation arrays across worker
# threads.
print("Calculating validation metrics serial...", flush=True)
serial_start = perf_counter()
serial_result = vl.ValidEngine(
    [vl.MetricAVM()], workers=1
).run(valid_case)
serial_seconds = perf_counter() - serial_start
print(f"Serial AVM: {serial_seconds:.3f} s", flush=True)

print(80*"-")
print("Calculating validation metrics in parallel (4 workers)...", flush=True)
parallel_start = perf_counter()
parallel_result = vl.ValidEngine(
    [vl.MetricAVM()], workers=4
).run(valid_case)
parallel_seconds = perf_counter() - parallel_start

print(80*"-")
print(f"Parallel AVM (4 workers): {parallel_seconds:.3f} s")
print(f"Combined solve: {serial_seconds + parallel_seconds:.3f} s")
print(f"Four-worker speedup: {serial_seconds / parallel_seconds:.2f}×")
print(80*"-")

# %%
# 5. Compare results and reshape to the image
# -------------------------------------------
#
# The AVM field retains independent simulation and experiment emissivity axes.
# Select case zero from both for the nominal comparison and take the maximum
# over both axes for the worst pair at each pixel. Reshape the point axis to
# image rows and columns for display.
serial_field = serial_result.metrics[observable_tag][vl.EMetricType.AVM]
parallel_field = parallel_result.metrics[observable_tag][vl.EMetricType.AVM]
serial_avm = serial_field.metric_vals[vl.EMetricOut.DISTANCE]
parallel_avm = parallel_field.metric_vals[vl.EMetricOut.DISTANCE]
np.testing.assert_array_equal(serial_avm, parallel_avm)

plot_opts = vl.ValidPlotOpts()
figure, axes = plt.subplots(
    2, 2, figsize=(7.25, 5.0), layout="constrained", dpi=plot_opts.resolution
)

avm_by_pixel = serial_avm[:, 0, 0]
nominal_avm = avm_by_pixel[:, 0, 0].reshape(CAMERA_HEIGHT, CAMERA_WIDTH)
worst_case_avm = avm_by_pixel.max(axis=(1, 2)).reshape(CAMERA_HEIGHT, CAMERA_WIDTH)

exp_emissivity_shift = exp_epistemic_temperature_shift[:, 1].reshape(
    CAMERA_HEIGHT, CAMERA_WIDTH
)
panels = (
    (ground_truth, "Ground truth"),
    (exp_emissivity_shift, "Largest experimental emissivity shift"),
    (nominal_avm, "AVM: nominal"),
    (worst_case_avm, "AVM: worst emissivity pair"),
)
plate_extent = (0.0, PLATE_WIDTH_M, 0.0, PLATE_HEIGHT_M)

for aa, (ii, tt) in zip(axes.flat, panels, strict=True):
    heatmap = aa.imshow(ii, origin="lower", extent=plate_extent)
    aa.set(title=tt, xlabel="x [m]", ylabel="y [m]")
    figure.colorbar(heatmap, ax=aa, label="Temperature [degC]")

fig_file = f"val_ex12_fig{next(fig_nums):02d}_ir_camera_avm.png"
figure.savefig(output_path / fig_file)

# %%
# .. image:: /_static/val_ex12_fig01_ir_camera_avm.png
#    :alt: IR camera temperature, emissivity shift, nominal AVM, and worst-pair AVM
#    :width: 100%
#    :align: center

if SHOW_PLOTS:
    plt.show()
else:
    plt.close(figure)
