"""Small prepared 2D plate datasets shared by validation examples."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyvale.data as dataset
import pyvale.valid as val


def save_diagnostic(fig: plt.Figure, name: str) -> Path:
    """Save and close a diagnostic figure in the normal example directory."""
    output_dir = Path.cwd() / "pyvale-output"
    output_dir.mkdir(exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def load_plate_case(
    *,
    transient: bool = False,
    epistemic: bool = False,
    offsets: bool = False,
) -> val.ValCase:
    """Load prepared SensorSim thermocouple and strain-gauge arrays.

    The packaged 2D plate simulations differ in conductivity and thermal
    expansion by a factor of 0.9. Data are (run, point, component, time).
    """
    with np.load(dataset.valid_data_dir() / "plate_validation_2d.npz") as data:
        coords = data["coords"].copy()
        times = data["times"].copy()
        nominal_temp = data["nominal_temperature"].copy()
        perturbed_temp = data["perturbed_temperature"].copy()
        nominal_strain = data["nominal_strain"].copy()
        perturbed_strain = data["perturbed_strain"].copy()

    geometry = val.ValGeometry(
        val.EValGeometry.POINT, coords, coordinate_units="mm", frame="plate"
    )
    evaluation_times = times if transient else None
    domain = val.ValDomain(
        {
            "temperature": val.ValObservable(
                geometry,
                val.EValOperator.SAMPLE,
                ("temperature",),
                evaluation_times,
                "degC",
                time_units="s" if transient else "",
            ),
            "strain": val.ValObservable(
                geometry,
                val.EValOperator.SAMPLE,
                ("strain_xx", "strain_yy", "strain_xy"),
                evaluation_times,
                "1",
                time_units="s" if transient else "",
            ),
        }
    )
    sim_arrays = {}
    exp_arrays = {}
    for name, nominal, changed in (
        ("temperature", nominal_temp, perturbed_temp),
        ("strain", nominal_strain, perturbed_strain),
    ):
        if transient:
            # (point, component, time, aleatory)
            exp_values = np.moveaxis(changed, 0, -1)
            sim_values = np.moveaxis(nominal, 0, -1)
            axes = (
                val.EValAxis.POINT,
                val.EValAxis.COMPONENT,
                val.EValAxis.TIME,
                val.EValAxis.ALEATORY,
            )
        else:
            # Final physical time, with repeated virtual experiments as samples.
            exp_values = np.moveaxis(changed[:, :, :, -1], 0, -1)
            sim_values = np.moveaxis(nominal[:, :, :, -1], 0, -1)
            axes = (val.EValAxis.POINT, val.EValAxis.COMPONENT, val.EValAxis.ALEATORY)
        if epistemic:
            changed_values = (
                np.moveaxis(changed, 0, -1)
                if transient
                else np.moveaxis(changed[:, :, :, -1], 0, -1)
            )
            sim_values = np.stack((sim_values, changed_values), axis=-2)
            axes = (*axes[:-1], val.EValAxis.EPISTEMIC_TRANS, val.EValAxis.ALEATORY)
        bound = None
        if offsets:
            magnitude = 0.5 if name == "temperature" else 4.0e-6
            if name == "temperature" and transient:
                # Calibration bounds vary by sensor position and evaluation time.
                spatial = coords[:, 0, None] / 100.0
                temporal = times[None, :] / times[-1]
                lower = magnitude * (1.0 + 0.2 * spatial + 0.2 * temporal)
                upper = 1.5 * magnitude * (1.0 + 0.1 * spatial + 0.3 * temporal)
                bound = val.EpistemicOffsets(
                    lower=lower,
                    upper=upper,
                    axes=(val.EValAxis.POINT, val.EValAxis.TIME),
                )
            else:
                bound = val.EpistemicOffsets(lower=magnitude, upper=1.5 * magnitude)
        exp_axes = (val.EValAxis.POINT, val.EValAxis.COMPONENT)
        if transient:
            exp_axes = (*exp_axes, val.EValAxis.TIME)
        exp_axes = (*exp_axes, val.EValAxis.ALEATORY)
        sim_arrays[name] = val.ValArray(sim_values, axes)
        exp_arrays[name] = val.ValArray(exp_values, exp_axes, bound)

    return val.ValCase(domain, val.ValData(sim_arrays), val.ValData(exp_arrays))
