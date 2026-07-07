from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pyvista as pv

from pyvale import mooseherder, sensorsim
from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import (
    BoundaryConditions,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.hardening import LinearHardening
from pyvale.vfm.identification import Identification, IdentificationPhase
from pyvale.vfm.metricsbvf import SensitivityBasedVirtualFieldsMetric
from pyvale.vfm.objectivefuncvector import VectorFirstResultPassthrough
from pyvale.vfm.optimiserleastsquares import LeastSquares
from pyvale.vfm.spatialparamhomogeneous import (
    HomogeneousSpatialParameterisation,
)
from pyvale.vfm.vfm import run_identification

PYVALE_ROOT = Path(__file__).resolve().parent.parent.parent
VFMVERIF_ROOT = PYVALE_ROOT.parent / "vfmverif_meshref_1"

EXODUS_FILE_NAME = "out_hole2d_plas_32f.e"
GRID_DIVS = 101

PLATE_WIDTH = 25e-3     # m
PLATE_HEIGHT = 35e-3    # m
PLATE_THICKNESS = 1e-3  # m

# Known homogeneous constitutive parameters used to generate the FE data.
KNOWN_PARAMETERS = {
    "elastic_modulus": 200_000.0,  # MPa
    "poissons_ratio": 0.3,
    "yield_strength": 200.0,       # MPa
    "hardening_modulus": 1_000.0,  # MPa
}

# Plot toggles for each stage of the test.
PLOT_STRESS_RECON_ABS_DIFF = True
PLOT_STRESS_RECON_ABS_PERC_DIFF = True
PLOT_METRIC_IDENTIFIED_DIFF = True
PLOT_IDENTIFICATION_DIFF = True

STRESS_COMPONENT_LABELS = ("xx", "yy", "xy")


def _rms(array: npt.NDArray[np.float64]) -> float:
    return float(np.sqrt(np.nanmean(np.square(array))))


def _root_mean_square_percentage_error(
    predicted: npt.NDArray[np.float64],
    known: npt.NDArray[np.float64],
) -> float:
    percentage_error = (predicted - known) / known * 100.0
    return float(np.sqrt(np.nanmean(np.square(percentage_error))))


def _plot_stress_abs_diff(
    x_grid: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    abs_diff: npt.NDArray[np.float64],
) -> None:
    """Plot the abs difference of each stress component at a single timestep."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, label, component in zip(
        axes, STRESS_COMPONENT_LABELS, range(3), strict=True
    ):
        field = abs_diff[component, :, :]
        image = ax.pcolormesh(x_grid, y_grid, field)
        fig.colorbar(image, ax=ax, label="|calc - FE| [MPa]")
        ax.set_title(f"stress_{label} abs diff")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.invert_yaxis()
    plt.show()

def _plot_stress_abs_perc_diff(
    x_grid: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    abs_perc_diff: npt.NDArray[np.float64],
) -> None:
    """Plot the absolute percentage difference of each stress component."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, label, component in zip(
        axes, STRESS_COMPONENT_LABELS, range(3), strict=True
    ):
        field = abs_perc_diff[component, :, :]
        image = ax.pcolormesh(x_grid, y_grid, field)
        fig.colorbar(image, ax=ax, label="|calc - FE| / |FE| [%]")
        image.set_clim(np.nanpercentile(field, 5), np.nanpercentile(field, 95))
        ax.set_title(f"stress_{label} abs % diff")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.invert_yaxis()
    plt.show()

def _plot_metric_virtual_work(
    internal_virtual_work_a: npt.NDArray[np.float64],
    external_virtual_work_a: npt.NDArray[np.float64],
    internal_virtual_work_b: npt.NDArray[np.float64],
    external_virtual_work_b: npt.NDArray[np.float64],
    label_a: str,
    label_b: str,
    sbvf_labels: tuple[str, ...],
) -> None:
    """Compare the internal/external virtual work of two metric evaluations.

    Each virtual work array has shape (num_virtual_fields, timesteps). One row
    of plots is drawn per SBVF, showing the IVW, EVW, abs difference and
    percentage difference between the two evaluations. Each SBVF corresponds to
    the single degree of freedom of one homogeneous constitutive parameter, so
    ``sbvf_labels`` names the parameter driving each row.
    """
    num_virtual_fields = internal_virtual_work_a.shape[0]

    # Figure 1: per-SBVF comparison of IVW and EVW between the two evaluations.
    fig_work, axes = plt.subplots(
        num_virtual_fields,
        4,
        figsize=(18, 3.5 * num_virtual_fields),
        constrained_layout=True,
        squeeze=False,
    )

    # Figure 2: per-SBVF comparison of the PVW residual |IVW - EVW| between the
    # two evaluations. Shown at the same time as figure 1.
    fig_residual, residual_axes = plt.subplots(
        num_virtual_fields,
        3,
        figsize=(13.5, 3.5 * num_virtual_fields),
        constrained_layout=True,
        squeeze=False,
    )

    for vf in range(num_virtual_fields):
        ivw_a = internal_virtual_work_a[vf]
        ivw_b = internal_virtual_work_b[vf]
        evw_a = external_virtual_work_a[vf]
        evw_b = external_virtual_work_b[vf]

        ivw_abs_diff = np.abs(ivw_b - ivw_a)
        evw_abs_diff = np.abs(evw_b - evw_a)
        # Guard against division by zero (e.g. zero virtual work at the first
        # timestep), leaving those points as NaN so they are skipped in the plot.
        ivw_percentage_diff = np.divide(
            ivw_abs_diff * 100.0,
            np.abs(ivw_a),
            out=np.full_like(ivw_abs_diff, np.nan),
            where=ivw_a != 0.0,
        )
        evw_percentage_diff = np.divide(
            evw_abs_diff * 100.0,
            np.abs(evw_a),
            out=np.full_like(evw_abs_diff, np.nan),
            where=evw_a != 0.0,
        )

        # PVW residual magnitude |IVW - EVW| for each evaluation, and the
        # residual as a percentage of EVW, separately for each evaluation.
        ivw_evw_diff_a = np.abs(ivw_a - evw_a)
        ivw_evw_diff_b = np.abs(ivw_b - evw_b)
        ivw_evw_percentage_diff_a = np.divide(
            ivw_evw_diff_a * 100.0,
            np.abs(evw_a),
            out=np.full_like(evw_a, np.nan),
            where=evw_a != 0.0,
        )
        ivw_evw_percentage_diff_b = np.divide(
            ivw_evw_diff_b * 100.0,
            np.abs(evw_b),
            out=np.full_like(evw_b, np.nan),
            where=evw_b != 0.0,
        )

        sbvf_label = sbvf_labels[vf]

        axes[vf, 0].plot(ivw_a, marker=".", label=label_a)
        axes[vf, 0].plot(ivw_b, marker=".", label=label_b)
        axes[vf, 0].set_title(f"{sbvf_label} IVW")
        axes[vf, 0].set_ylabel("internal virtual work")
        axes[vf, 0].legend()

        axes[vf, 1].plot(evw_a, marker=".", label=label_a)
        axes[vf, 1].plot(evw_b, marker=".", label=label_b)
        axes[vf, 1].set_title(f"{sbvf_label} EVW")
        axes[vf, 1].set_ylabel("external virtual work")
        axes[vf, 1].legend()

        axes[vf, 2].plot(ivw_abs_diff, marker=".", label="IVW")
        axes[vf, 2].plot(evw_abs_diff, marker=".", label="EVW")
        axes[vf, 2].set_title(f"{sbvf_label} abs diff")
        axes[vf, 2].set_ylabel(f"|{label_b} - {label_a}|")
        axes[vf, 2].legend()

        axes[vf, 3].plot(ivw_percentage_diff, marker=".", label="IVW")
        axes[vf, 3].plot(evw_percentage_diff, marker=".", label="EVW")
        axes[vf, 3].set_title(f"{sbvf_label} percentage diff")
        axes[vf, 3].set_ylabel("% diff")
        axes[vf, 3].legend()

        # Evaluation a (e.g. known) in blue, b (e.g. calc) in orange; IVW solid,
        # EVW dashed.
        residual_axes[vf, 0].plot(
            ivw_a, marker=".", color="blue", linestyle="-", label=f"IVW {label_a}"
        )
        residual_axes[vf, 0].plot(
            evw_a, marker=".", color="blue", linestyle="--", label=f"EVW {label_a}"
        )
        residual_axes[vf, 0].plot(
            ivw_b, marker=".", color="orange", linestyle="-", label=f"IVW {label_b}"
        )
        residual_axes[vf, 0].plot(
            evw_b, marker=".", color="orange", linestyle="--", label=f"EVW {label_b}"
        )
        residual_axes[vf, 0].set_title(f"{sbvf_label} IVW & EVW")
        residual_axes[vf, 0].set_ylabel("virtual work")
        residual_axes[vf, 0].legend()

        residual_axes[vf, 1].plot(ivw_evw_diff_a, marker=".", label=label_a)
        residual_axes[vf, 1].plot(ivw_evw_diff_b, marker=".", label=label_b)
        residual_axes[vf, 1].set_title(f"{sbvf_label} |IVW - EVW|")
        residual_axes[vf, 1].set_ylabel("|IVW - EVW|")
        residual_axes[vf, 1].legend()

        residual_axes[vf, 2].plot(ivw_evw_percentage_diff_a, marker=".", label=label_a)
        residual_axes[vf, 2].plot(ivw_evw_percentage_diff_b, marker=".", label=label_b)
        residual_axes[vf, 2].set_title(f"{sbvf_label} |IVW - EVW| percentage diff")
        residual_axes[vf, 2].set_ylabel("% diff (IVW vs EVW)")
        residual_axes[vf, 2].legend()

        for column in range(4):
            axes[vf, column].set_xlabel("timestep")
        for column in range(3):
            residual_axes[vf, column].set_xlabel("timestep")

    plt.show()


def _plot_identification_diff(
    x_grid: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    identified_maps: dict[str, npt.NDArray[np.float64]],
    known_maps: dict[str, npt.NDArray[np.float64]],
) -> None:
    """Plot the difference between the identified and known parameter maps."""
    fig, axes = plt.subplots(1, len(known_maps), figsize=(16, 4), constrained_layout=True)
    for ax, param_name in zip(axes, known_maps, strict=True):
        field = identified_maps[param_name] - known_maps[param_name]
        image = ax.pcolormesh(x_grid, y_grid, field)
        fig.colorbar(image, ax=ax, label="identified - known")
        ax.set_title(param_name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.invert_yaxis()
    plt.show()


def _load_sim_data_to_grid(
    exodus_file_name: str,
    component_keys: tuple[str, ...],
    grid_divs: int,
) -> tuple[
    npt.NDArray[np.float64],  # x_grid, shape (x, y, z)
    npt.NDArray[np.float64],  # y_grid, shape (x, y, z)
    npt.NDArray[np.float64],  # grid_data, shape (x, y, z, components, timesteps)
    npt.NDArray[np.float64],  # force, shape (timesteps)
    npt.NDArray[np.float64],  # time, shape (timesteps)
]:
    exodus_file_path = VFMVERIF_ROOT / exodus_file_name

    sim_data = mooseherder.ExodusLoader(exodus_file_path).load_all_sim_data()

    def grid_inner_vec(lower: float, upper: float, num_divs: int) -> np.ndarray:
        step = (upper - lower) / num_divs
        start = lower + (step / 2)
        stop = upper - (step / 2)
        return np.linspace(start, stop, num_divs)

    x_vec = grid_inner_vec(PLATE_WIDTH / 2, -PLATE_WIDTH / 2, grid_divs)
    y_vec = (
        grid_inner_vec(PLATE_HEIGHT / 2, -PLATE_HEIGHT / 2, grid_divs)
        + PLATE_HEIGHT / 2
    )
    z_vec = np.full((1,), 0.0, dtype=np.float64)

    (x_grid, y_grid, z_grid) = np.meshgrid(x_vec, y_vec, z_vec, indexing="ij")

    interp_grid = np.stack([x_grid, y_grid, z_grid], axis=0)
    spatial_grid_shape = interp_grid.shape[1:]
    interp_points = interp_grid.reshape(3, -1).T

    pyvista_interp = sensorsim.simdata_to_pyvista_interp(
        sim_data,
        component_keys,
        sensorsim.EDim.TWOD,
    )
    pv_points = pv.PolyData(interp_points)
    sample_data = pv_points.sample(pyvista_interp)

    invalid = ~sample_data["vtkValidPointMask"].astype(bool)

    n_comps = len(component_keys)
    (n_sensors, n_time_steps) = np.array(sample_data[component_keys[0]]).shape
    sample_at_sim_time = np.empty((n_sensors, n_comps, n_time_steps))

    for ii, cc in enumerate(component_keys):
        data_mat = np.array(sample_data[cc])
        data_mat[invalid, :] = np.nan
        sample_at_sim_time[:, ii, :] = data_mat

    # Target: (Nx, Ny, Nz, n_comps, n_time_steps)
    final_shape = spatial_grid_shape + (n_comps, n_time_steps)
    grid_data = sample_at_sim_time.reshape(final_shape)

    return (
        x_grid,
        y_grid,
        grid_data,
        sim_data.glob_vars["react_y_top"],
        sim_data.time,
    )


def test_end_to_end() -> None:
    # ------------------------------------------------------------------
    # Setup: extract 2d strain and stress components from the .e file and
    # build the grid data and identification objects.
    # ------------------------------------------------------------------
    print("Loading data...")
    component_keys = (
        "strain_xx",
        "strain_yy",
        "strain_xy",
        "stress_xx",
        "stress_yy",
        "stress_xy",
    )

    (x_grid, y_grid, grid_data, force, time) = _load_sim_data_to_grid(
        EXODUS_FILE_NAME,
        component_keys,
        GRID_DIVS,
    )

    # Reshape and flip data to match our conventions.
    # remove redundant z component
    x_grid = x_grid[:, :, 0]  # shape: (x, y)
    y_grid = y_grid[:, :, 0]  # shape: (x, y)
    grid_data = grid_data[:, :, 0, :, :]  # shape: (x, y, components, timesteps)

    # reshape the grid and data to use our conventions
    x_grid = x_grid.transpose(1, 0)  # shape: (y, x)
    y_grid = y_grid.transpose(1, 0)  # shape: (y, x)
    grid_data = grid_data.transpose(3, 2, 1, 0)  # shape: (timesteps, components, y, x)

    # x increases with column number, is constant in each column, always positive
    x_grid = np.fliplr(x_grid)
    x_grid += np.nanmax(x_grid)
    grid_data = np.flip(grid_data, axis=2)

    # y increases with row number, is constant in each row, always positive
    y_grid = np.flipud(y_grid)
    grid_data = np.flip(grid_data, axis=3)

    # convert stress components from Pa to MPa
    grid_data[:, 3:6, :, :] *= 1e-6

    strain = grid_data[:, 0:3, :, :]  # shape: (timesteps, 3, y, x) [xx, yy, xy]
    stress_fe = grid_data[:, 3:6, :, :]  # shape: (timesteps, 3, y, x) [xx, yy, xy]

    specimen_mask = ~np.isnan(strain[0, 0, :, :])

    grid_element_area = (
        (x_grid[0, 1] - x_grid[0, 0]) * (y_grid[1, 0] - y_grid[0, 0])
    )

    specimen_geometry = SpecimenGeometry(
        x_grid,
        y_grid,
        specimen_mask,
        PLATE_THICKNESS,
        np.full_like(x_grid, grid_element_area, dtype=np.float64),
    )

    # seems to be an issue with FE input force data being 1000x too large
    force *= 1e-3

    boundary_conditions = BoundaryConditions(
        EdgeConditions(
            min_x_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Free),
            max_x_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Free),
            min_y_edge=Edge(x=EEdgeCondition.Fixed, y=EEdgeCondition.Fixed),
            max_y_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Traction),
        ),
        np.column_stack((np.zeros_like(force), force)),
    )

    experiment_data = ExperimentData(
        strain,
        specimen_geometry,
        boundary_conditions,
        time,
    )

    constitutive_law = IsotropicVonMisesElastoplasticity(LinearHardening())

    parameter_map_size = np.array([GRID_DIVS, GRID_DIVS], dtype=np.uint32)

    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            450_000, 100_000, 500_000, parameter_map_size
        ),
        "poissons_ratio": ConstitutiveParameter(
            0.45, 0.1, 0.5, parameter_map_size
        ),
        "yield_strength": ConstitutiveParameter(
            800, 100, 1000, parameter_map_size
        ),
        "hardening_modulus": ConstitutiveParameter(
            7000, 500, 10_000, parameter_map_size
        ),
    }

    metric = SensitivityBasedVirtualFieldsMetric(
        experiment_data.specimen_geometry.x,
        experiment_data.specimen_geometry.y,
        experiment_data.specimen_geometry.region_of_interest,
        experiment_data.boundary_conditions.edge_conditions,
        np.array([15, 15]),
    )

    phases = [
        IdentificationPhase(
            {
                "elastic_modulus": HomogeneousSpatialParameterisation(),
                "poissons_ratio": HomogeneousSpatialParameterisation(),
                "yield_strength": HomogeneousSpatialParameterisation(),
                "hardening_modulus": HomogeneousSpatialParameterisation(),
            },
            [metric],
            VectorFirstResultPassthrough(),
            LeastSquares(),
        )
    ]

    identification = Identification(constitutive_law, parameters, phases)

    # Known homogeneous constitutive parameter maps.
    known_parameter_maps = {
        name: np.full((GRID_DIVS, GRID_DIVS), value)
        for name, value in KNOWN_PARAMETERS.items()
    }

    # ------------------------------------------------------------------
    # Test the stress reconstruction: reconstruct stress from the known
    # homogeneous parameters and compare against the FE stress.
    # ------------------------------------------------------------------
    print("Reconstructing stress...")
    stress_calc = constitutive_law.calculate_stress(strain, known_parameter_maps)

    # abs difference between calculated and known (FE) stress at final timestep
    stress_abs_diff = np.abs(stress_calc[-1] - stress_fe[-1])  # shape: (3, y, x)

    if PLOT_STRESS_RECON_ABS_DIFF:
        _plot_stress_abs_diff(x_grid, y_grid, stress_abs_diff)

    stress_abs_perc_diff = np.full_like(stress_fe[-1], np.nan, dtype=np.float64)
    valid = np.abs(stress_fe[-1]) > 0.01 #avoid division by zero
    stress_abs_perc_diff[valid] = (
        np.abs(stress_calc[-1][valid] - stress_fe[-1][valid])
        / np.abs(stress_fe[-1][valid])
    ) * 100.0  # shape: (3, y, x)

    if PLOT_STRESS_RECON_ABS_PERC_DIFF:
        _plot_stress_abs_perc_diff(x_grid, y_grid, stress_abs_perc_diff)

    stress_abs_diff_mean = float(np.nanmean(stress_abs_diff))
    stress_abs_diff_max = float(np.nanmax(stress_abs_diff))
    stress_abs_diff_rms = _rms(stress_abs_diff)

    print(f"stress recon abs diff mean [MPa] = {stress_abs_diff_mean:.6f}")
    print(f"stress recon abs diff max  [MPa] = {stress_abs_diff_max:.6f}")
    print(f"stress recon abs diff rms  [MPa] = {stress_abs_diff_rms:.6f}")

    # The calculated stress reconstruction should be close to the known FE
    # stress, so the abs difference statistics should be small relative to the
    # stress magnitude (~hundreds of MPa).
    assert stress_abs_diff_mean < 0.5
    assert stress_abs_diff_max < 10.0
    assert stress_abs_diff_rms < 1.0

    # ------------------------------------------------------------------
    # Run the identification with all constitutive parameters set to
    # homogeneous.
    # ------------------------------------------------------------------
    print("Running identification...")
    identified_parameters = run_identification(experiment_data, identification)

    # Copy the internal/external virtual work from the metric's final evaluation
    # during the identification (i.e. at the identified parameters).
    ivw_identified = metric._internal_virtual_work.copy()
    evw_identified = metric._external_virtual_work.copy()

    identified_maps = {
        name: param.value for name, param in identified_parameters.items()
    }

    for name, param in identified_parameters.items():
        print(f"{name} = {np.nanmean(param.value):.6f}")

    # ------------------------------------------------------------------
    # Test the performance of the metric: compare the SBVF metric evaluated with
    # the known (FE) stress against the metric at the identified parameters.
    # Both should give a similar residual vector. This only makes sense once the
    # identification has produced parameters to compare against.
    # ------------------------------------------------------------------
    print("Evaluating metric...")
    metric_spatial_parameterisations = {
        name: HomogeneousSpatialParameterisation()
        for name in KNOWN_PARAMETERS
    }
    for name, spatial_parameterisation in metric_spatial_parameterisations.items():
        spatial_parameterisation.update_from_constitutive_parameter(
            ConstitutiveParameter(
                known_parameter_maps[name],
                parameters[name].lower_bound,
                parameters[name].upper_bound,
            )
        )

    metric.evaluate(
        stress_fe,
        constitutive_law,
        parameter_map_size,
        metric_spatial_parameterisations,
        experiment_data,
    )
    # Copy the internal/external virtual work for the known-stress evaluation.
    ivw_known = metric._internal_virtual_work.copy()
    evw_known = metric._external_virtual_work.copy()

    # Each SBVF corresponds to the single dof of one homogeneous parameter, in
    # the order the parameters are defined.
    sbvf_labels = tuple(name.replace("_", " ") for name in KNOWN_PARAMETERS)

    if PLOT_METRIC_IDENTIFIED_DIFF:
        _plot_metric_virtual_work(
            ivw_known, evw_known, ivw_identified, evw_identified,
            "known", "identified", sbvf_labels,
        )

    # Relative RMS difference of the internal/external virtual work between the
    # known-stress evaluation and the identified parameters, normalised by the
    # known-stress scale. The residual (IVW - EVW) itself is not compared because
    # the identification drives it to ~0 by construction, whereas the known
    # residual is non-zero.
    ivw_relative_diff = _rms(ivw_identified - ivw_known) / _rms(ivw_known)
    evw_relative_diff = _rms(evw_identified - evw_known) / _rms(evw_known)

    print(f"metric IVW relative diff (known vs identified) = {ivw_relative_diff:.6f}")
    print(f"metric EVW relative diff (known vs identified) = {evw_relative_diff:.6f}")

    # The internal/external virtual work at the identified parameters should be
    # close to those from the known (FE) stress.
    assert ivw_relative_diff < 0.05
    assert evw_relative_diff < 0.05

    # ------------------------------------------------------------------
    # Test the result of the identification: compare the identified parameter
    # maps against the known parameter maps.
    # ------------------------------------------------------------------
    if PLOT_IDENTIFICATION_DIFF:
        _plot_identification_diff(
            x_grid, y_grid, identified_maps, known_parameter_maps
        )

    # Per-parameter tolerances on the RMS of the absolute difference. The
    # hardening modulus is only weakly sensitive to the virtual fields and so
    # is identified less accurately than the other parameters.
    abs_diff_rms_tolerances = {
        "elastic_modulus": 400.0,
        "poissons_ratio": 1e-3,
        "yield_strength": 1.0,
        "hardening_modulus": 250.0,
    }

    for name in KNOWN_PARAMETERS:
        abs_diff = np.abs(identified_maps[name] - known_parameter_maps[name])
        abs_diff_rms = _rms(abs_diff)
        rmspe = _root_mean_square_percentage_error(
            identified_maps[name], known_parameter_maps[name]
        )
        print(
            f"{name}: abs diff rms = {abs_diff_rms:.6f}, rmspe = {rmspe:.6f} %"
        )

        # The identified parameters should be close to the known parameters.
        assert abs_diff_rms < abs_diff_rms_tolerances[name]
        assert rmspe < 20.0

test_end_to_end()
