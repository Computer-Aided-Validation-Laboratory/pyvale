# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Choose sample times to capture a heating peak
=============================================
"""

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import pyvale.dataio as io
import pyvale.sensoropt as opt
import pyvale.sensorsim as sens
from pyvale.examples.sensopt.thermal_pulse import build_thermal_pulse

# %%
# 1. Install a sensor at the centre of the heated region
# ------------------------------------------------------

plate: io.SimData = build_thermal_pulse()
array: sens.SensorsPoint = sens.SensorFactory.scalar_point(
    plate,
    sens.SensorData(positions=np.array([[3.75, 4.5, 0.0]])),
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.temperature(),
)

# %%
# 2. Compare irregular timestamps and regular frequencies at three readings
# -------------------------------------------------------------------------

native_times = cast(npt.NDArray[np.float64], plate.time)
irregular_times = opt.gen_sample_time_combinations(
    candidate_times=native_times[[0, 12, 16, 18, 22, 28, 40]],
    num_samples=3,
)
# These frequencies all produce three samples in the acquisition window.
frequencies: npt.NDArray[np.float64] = 1.0 / np.array([0.39, 0.43, 0.45, 0.48])
regular_times = opt.gen_sample_times_from_freqs(
    frequencies=frequencies, time_window=(0.0, 1.0)
)
schedule_sets: dict[str, tuple[npt.NDArray[np.float64], ...]] = {
    "Irregular timestamps": irregular_times,
    "Regular frequencies": regular_times,
}

results: dict[str, opt.OptResult] = {}
for name, schedules in schedule_sets.items():
    result = opt.SensorOptEngine(
        sensor_arrays={"temperature": array},
        design_space=opt.DesignSpace(sample_times={"temperature": schedules}),
        objectives=(opt.ObjSignalPeakAbs(sensor_keys=("temperature",)),),
        optimiser=opt.OptFixedLayout(),
        seed=123,
    ).optimise()
    results[name] = result
    selected = result.sensor_data["temperature"]
    print(
        f"{name}: sample times {selected.sample_times}, peak {-result.objective_values[0]:.3f}"
    )

# %%
# 3. Plot the measured samples over the native simulation trace
# ------------------------------------------------------------

output_path: Path = Path.cwd() / "pyvale-output" / "sensopt_ex10_peak_sample_times"
output_path.mkdir(parents=True, exist_ok=True)
figure, axes = plt.subplots(layout="constrained")
axes.plot(native_times, array.get_truth()[0, 0], label="Simulation trace")
for name, result in results.items():
    selected = result.sensor_data["temperature"]
    sampled = sens.SensorsPoint(selected, array.get_field()).get_truth()[0, 0]
    times = cast(npt.NDArray[np.float64], selected.sample_times)
    axes.scatter(times, sampled, label=name)
axes.set(xlabel="Time [s]", ylabel="Temperature")
axes.legend()
figure.savefig(output_path / "peak_samples.png", dpi=160)
plt.close(figure)
