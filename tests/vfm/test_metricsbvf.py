import numpy as np
from load_sim_data import load_force, load_strain, load_stress, load_timesteps
from plots import (
    _plot_metric_virtual_work,
)
from utils import rms

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

EXODUS_FILE_NAME = "out_hole2d_plas_32f.e"
GRID_DIVS = 101

PLATE_THICKNESS = 1e-3 # m

KNOWN_PARAMETERS = {
    "elastic_modulus": 200_000.0,  # MPa
    "poissons_ratio": 0.3,
    "yield_strength": 200.0,       # MPa
    "hardening_modulus": 1_000.0,  # MPa
}

PLOT_METRIC_IDENTIFIED_DIFF = False


# Compute virtual fields using fe model stress and known parameter
# values, then use those vfs for any further metric evaluation
def test_sbvf_metric_with_vfs_locked():
    (_, _, stress_fe) = load_stress(EXODUS_FILE_NAME, GRID_DIVS)

    experiment_data = _setup_experiment_data()
    ident = _setup_identification(experiment_data)

    sbvf_metric = ident.phases[0].metrics[0]

    # Known homogeneous constitutive parameter maps.
    known_parameter_maps = {
        name: np.full((GRID_DIVS, GRID_DIVS), value)
        for name, value in KNOWN_PARAMETERS.items()
    }

    # Evaluate the sbvf metric and generate virtual fields
    # ahead of identification with known values of parameters
    # and stress
    metric_spatial_parameterisations = {
        name: HomogeneousSpatialParameterisation()
        for name in KNOWN_PARAMETERS
    }

    for name, spatial_parameterisation in metric_spatial_parameterisations.items():
        spatial_parameterisation.update_from_constitutive_parameter(
            ConstitutiveParameter(
                known_parameter_maps[name],
                ident.parameters[name].lower_bound,
                ident.parameters[name].upper_bound,
            )
        )

    parameter_map_size = np.array([GRID_DIVS, GRID_DIVS], dtype=np.uint32)

    sbvf_metric.initialise(experiment_data)

    sbvf_metric.evaluate(
        stress_fe,
        ident.constitutive_law,
        parameter_map_size,
        metric_spatial_parameterisations,
        experiment_data,
    )

    # Save the internal/external virtual work for the known-stress evaluation.
    ivw_known = sbvf_metric._internal_virtual_work.copy()
    evw_known = sbvf_metric._external_virtual_work.copy()

    # Lock virtual fields for the following metric evaluations
    sbvf_metric._recompute_virtual_fields = False

    # ------------------------------------------------------------------
    # Run the identification with all constitutive parameters set to
    # homogeneous.
    # ------------------------------------------------------------------
    print("Running identification...")
    identified_parameters = run_identification(experiment_data, ident)

    # Copy the internal/external virtual work from the metric's final evaluation
    # during the identification (i.e. at the identified parameters).
    ivw_identified = sbvf_metric._internal_virtual_work.copy()
    evw_identified = sbvf_metric._external_virtual_work.copy()

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
    ivw_relative_diff = rms(ivw_identified - ivw_known) / rms(ivw_known)
    evw_relative_diff = rms(evw_identified - evw_known) / rms(evw_known)

    print(f"metric IVW relative diff (known vs identified) = {ivw_relative_diff:.6f}")
    print(f"metric EVW relative diff (known vs identified) = {evw_relative_diff:.6f}")

    # The internal/external virtual work at the identified parameters should be
    # close to those from the known (FE) stress.
    assert ivw_relative_diff < 0.05
    assert evw_relative_diff < 0.05

# Virtual fields are recomputed on every metric evaluation
def test_sbvf_metric_with_vfs_free():
    (_, _, stress_fe) = load_stress(EXODUS_FILE_NAME, GRID_DIVS)

    experiment_data = _setup_experiment_data()
    ident = _setup_identification(experiment_data)

    sbvf_metric = ident.phases[0].metrics[0]

    # Known homogeneous constitutive parameter maps.
    known_parameter_maps = {
        name: np.full((GRID_DIVS, GRID_DIVS), value)
        for name, value in KNOWN_PARAMETERS.items()
    }

    # Evaluate the sbvf metric and generate virtual fields
    # ahead of identification with known values of parameters
    # and stress
    metric_spatial_parameterisations = {
        name: HomogeneousSpatialParameterisation()
        for name in KNOWN_PARAMETERS
    }

    for name, spatial_parameterisation in metric_spatial_parameterisations.items():
        spatial_parameterisation.update_from_constitutive_parameter(
            ConstitutiveParameter(
                known_parameter_maps[name],
                ident.parameters[name].lower_bound,
                ident.parameters[name].upper_bound,
            )
        )

    parameter_map_size = np.array([GRID_DIVS, GRID_DIVS], dtype=np.uint32)

    sbvf_metric.initialise(experiment_data)

    sbvf_metric.evaluate(
        stress_fe,
        ident.constitutive_law,
        parameter_map_size,
        metric_spatial_parameterisations,
        experiment_data,
    )

    # Save the internal/external virtual work for the known-stress evaluation.
    ivw_known = sbvf_metric._internal_virtual_work.copy()
    evw_known = sbvf_metric._external_virtual_work.copy()

    # ------------------------------------------------------------------
    # Run the identification with all constitutive parameters set to
    # homogeneous.
    # ------------------------------------------------------------------
    print("Running identification...")
    identified_parameters = run_identification(experiment_data, ident)

    # Copy the internal/external virtual work from the metric's final evaluation
    # during the identification (i.e. at the identified parameters).
    ivw_identified = sbvf_metric._internal_virtual_work.copy()
    evw_identified = sbvf_metric._external_virtual_work.copy()

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
    ivw_relative_diff = rms(ivw_identified - ivw_known) / rms(ivw_known)
    evw_relative_diff = rms(evw_identified - evw_known) / rms(evw_known)

    print(f"metric IVW relative diff (known vs identified) = {ivw_relative_diff:.6f}")
    print(f"metric EVW relative diff (known vs identified) = {evw_relative_diff:.6f}")

    # The internal/external virtual work at the identified parameters should be
    # close to those from the known (FE) stress.
    assert ivw_relative_diff < 0.05
    assert evw_relative_diff < 0.05


def _setup_experiment_data() -> ExperimentData:
    (x_grid, y_grid, strain) = load_strain(EXODUS_FILE_NAME, GRID_DIVS)
    force = load_force(EXODUS_FILE_NAME)
    timesteps = load_timesteps(EXODUS_FILE_NAME)

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
        force
    )

    return ExperimentData(
        strain,
        specimen_geometry,
        boundary_conditions,
        timesteps,
    )


def _setup_identification(experiment_data: ExperimentData) -> Identification:

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

    metric = SensitivityBasedVirtualFieldsMetric(np.array([15, 15]))

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

    return Identification(constitutive_law, parameters, phases)
