# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

import numpy as np

import pyvale.data as dataset
import pyvale.validation as vl


def load_plate_case(
    *,
    transient: bool = False,
    epistemic: bool = False,
    offsets: bool = False,
) -> vl.ValidCase:
    """Load prepared SensorSim thermocouple and strain-gauge arrays.

    The packaged 2D plate simulations differ in conductivity and thermal
    expansion by a factor of 0.9. Saved validation arrays have repeats last.
    """
    if transient:
        filename = (
            "plate_validation_2d.npz"
            if epistemic
            else "plate_validation_2d_nominal.npz"
        )
    else:
        filename = (
            "plate_validation_2d_steady.npz"
            if epistemic
            else "plate_validation_2d_steady_nominal.npz"
        )
    with np.load(dataset.valid_data_dir() / filename) as data:
        coords = data["coords"].copy()
        times = data["times"].copy() if transient else None
        sim_temp = data["sim_temp"].copy()
        exp_temp = data["exp_temp"].copy()
        sim_strain = data["sim_strain"].copy()
        exp_strain = data["exp_strain"].copy()

    evaluation_times = times if transient else None
    observables = {
        "temperature": vl.ValidObservable(
            components=("temperature",),
            geometry=vl.EValidGeometry.POINT,
            coords=coords,
            times=evaluation_times,
            phys_units="degC",
            coord_units="mm",
            coord_frame="plate",
            time_units="s" if transient else "",
        ),
        "strain": vl.ValidObservable(
            components=("strain_xx", "strain_yy", "strain_xy"),
            geometry=vl.EValidGeometry.POINT,
            coords=coords,
            times=evaluation_times,
            phys_units="1",
            coord_units="mm",
            coord_frame="plate",
            time_units="s" if transient else "",
        ),
    }
    sim_arrays = {}
    exp_arrays = {}
    for name, sim_values, exp_values in (
        ("temperature", sim_temp, exp_temp),
        ("strain", sim_strain, exp_strain),
    ):
        axes = (vl.EValidAxis.POINT, vl.EValidAxis.COMPONENT)
        if transient:
            axes = (*axes, vl.EValidAxis.TIME)
        if epistemic:
            axes = (*axes, vl.EValidAxis.EPISTEMIC_TRANS)
        axes = (*axes, vl.EValidAxis.ALEATORY)
        bound = None
        if offsets:
            magnitude = 0.5 if name == "temperature" else 4.0e-6
            if name == "temperature" and transient:
                # Calibration bounds vary by sensor position and evaluation time.
                spatial = coords[:, 0, None] / 100.0
                temporal = times[None, :] / times[-1]
                lower = magnitude * (1.0 + 0.2 * spatial + 0.2 * temporal)
                upper = 1.5 * magnitude * (1.0 + 0.1 * spatial + 0.3 * temporal)
                bound = vl.EpistemicOffsets(
                    lower=lower,
                    upper=upper,
                    axes=(vl.EValidAxis.POINT, vl.EValidAxis.TIME),
                )
            else:
                bound = vl.EpistemicOffsets(lower=magnitude, upper=1.5 * magnitude)
        exp_axes = (vl.EValidAxis.POINT, vl.EValidAxis.COMPONENT)
        if transient:
            exp_axes = (*exp_axes, vl.EValidAxis.TIME)
        exp_axes = (*exp_axes, vl.EValidAxis.ALEATORY)
        sim_arrays[name] = vl.ValidArray(sim_values, axes)
        exp_arrays[name] = vl.ValidArray(exp_values, exp_axes, bound)

    return vl.ValidCase(observables, sim_arrays, exp_arrays)
