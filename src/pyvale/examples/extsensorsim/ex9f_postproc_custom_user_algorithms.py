# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Extended Example 9f: Custom User-Defined Post-Processing Algorithms
================================================================================

In specialized experimental mechanics, validation metrics often require
proprietary domain-specific calculations. Using `ProcessCustom`, users can
seamlessly inject arbitrary Python functions into post-processing pipelines
and DAGs without modifying Pyvale internals.

In this example, we demonstrate:
1. Simulating simultaneous thermal (thermocouple) and displacement (LVDT)
   transducers on a heating rod.
2. Defining a custom functional to calculate apparent thermal expansion
   coefficient
   :math:`\\alpha_{\\text{eff}}(t) = \\frac{\\Delta L / L_0}{\\Delta T(t)}`.
3. Wrapping the algorithm in `ProcessCustom` and executing through
   `PostProcessGraph`.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pyvale.sensorsim.measurementdata import MeasurementData
from pyvale.sensorsim.postprocessfilters import ProcessFilterSavitzkyGolay
from pyvale.sensorsim.postprocessderived import ProcessCustom
from pyvale.sensorsim.postprocessgraph import PostProcessGraph


def calc_effective_cte(
    inputs: dict[str, MeasurementData],
    initial_length: float = 100.0,
    initial_temp: float = 20.0,
) -> MeasurementData:
    """User-defined algorithm computing apparent thermal expansion."""
    temp_data = inputs["temp_smooth"]
    disp_data = inputs["disp_smooth"]

    temp_vals = temp_data.values
    disp_vals = disp_data.values

    delta_t = temp_vals - initial_temp
    delta_t_safe = np.where(np.abs(delta_t) < 0.5, 0.5, delta_t)

    # alpha = (dL / L0) / dT in microstrain / deg C (1e-6 / C)
    strain_thermal = disp_vals / initial_length
    cte_eff = (strain_thermal / delta_t_safe) * 1e6

    return MeasurementData(
        values=cte_eff,
        sample_times=temp_data.sample_times,
        positions=disp_data.positions,
        components=("alpha_eff",),
        units="μm/(m·K)",
    )


def main(show_plots: bool = False) -> None:
    # --------------------------------------------------------------------------
    # 1. Generate Synthetic Thermomechanical Heating Data
    L0 = 100.0  # initial rod length (mm)
    T0 = 20.0  # ambient temperature (deg C)
    true_alpha = 23.0  # aluminum CTE: 23e-6 1/K

    times = np.linspace(0.0, 10.0, 200)
    # Temperature heating curve T(t) = T0 + 150*(1 - exp(-t/3))
    t_curve = T0 + 150.0 * (1.0 - np.exp(-times / 3.0))
    # Thermal expansion dL(t) = L0 * true_alpha * (T(t) - T0)
    disp_true = L0 * (true_alpha * 1e-6) * (t_curve - T0)

    # Add realistic sensor errors
    rng = np.random.default_rng(55)
    t_meas = t_curve + rng.normal(0.0, 1.2, size=t_curve.shape)
    disp_meas = disp_true + rng.normal(0.0, 0.003, size=disp_true.shape)

    temp_data = MeasurementData(
        values=t_meas[np.newaxis, np.newaxis, :],
        sample_times=times,
        positions=np.zeros((1, 3)),
        components=("temperature",),
        units="°C",
    )

    disp_data = MeasurementData(
        values=disp_meas[np.newaxis, np.newaxis, :],
        sample_times=times,
        positions=np.zeros((1, 3)),
        components=("displacement",),
        units="mm",
    )

    # --------------------------------------------------------------------------
    # 2. Build DAG with Pre-Smoothing and Custom CTE Calculation
    dag = PostProcessGraph()
    dag.add_processor(
        "temp_smooth",
        ProcessFilterSavitzkyGolay(
            source="temp_raw", window_length=15, polyorder=2
        ),
    )
    dag.add_processor(
        "disp_smooth",
        ProcessFilterSavitzkyGolay(
            source="disp_raw", window_length=15, polyorder=2
        ),
    )
    dag.add_processor(
        "cte_calc",
        ProcessCustom(
            sources={"temp": "temp_smooth", "disp": "disp_smooth"},
            func=calc_effective_cte,
            initial_length=L0,
            initial_temp=T0,
            output_components=("alpha_eff",),
        ),
    )

    results = dag.execute({"temp_raw": temp_data, "disp_raw": disp_data})
    cte_result = results["cte_calc"]

    # --------------------------------------------------------------------------
    # 3. Plotting & Verification
    out_dir = Path("pyvale-output/extsensorsim")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # Panel 1: Temperature
    axes[0].plot(times, t_curve, "k--", label="True Temperature")
    axes[0].plot(times, t_meas, "r.", alpha=0.3, label="Noisy Thermocouple")
    axes[0].plot(
        times,
        results["temp_smooth"].values[0, 0],
        "r-",
        lw=2,
        label="Filtered Temp",
    )
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("User-Defined Post-Processing: Apparent CTE Calculation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    # Panel 2: Thermal Expansion Displacement
    axes[1].plot(times, disp_true, "k--", label="True Expansion dL")
    axes[1].plot(times, disp_meas, "b.", alpha=0.3, label="Noisy LVDT")
    axes[1].plot(
        times,
        results["disp_smooth"].values[0, 0],
        "b-",
        lw=2,
        label="Filtered dL",
    )
    axes[1].set_ylabel("Displacement (mm)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    # Panel 3: Effective CTE
    valid_idx = times > 0.5
    axes[2].axhline(
        true_alpha, color="k", linestyle="--", label="True CTE (23.0 μm/(m·K))"
    )
    axes[2].plot(
        times[valid_idx],
        cte_result.values[0, 0, valid_idx],
        "g-",
        lw=2,
        label="Derived Apparent CTE",
    )
    axes[2].set_ylabel("CTE (μm/(m·K))")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(15.0, 30.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "ext_ex9f_custom_user_postproc.png", dpi=150)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main(show_plots=True)
