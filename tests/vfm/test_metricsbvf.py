from pathlib import Path

import numpy as np
import pytest
from plots import (
    plot_metric_virtual_work,
)
from rms import rms

from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.hardening import HardeningLinear
from pyvale.vfm.identification import run_identification
from pyvale.vfm.identificationconfig import (
    IdentificationConfig,
    IdentificationPhase,
)
from pyvale.vfm.metricsbvf import MetricSBVF
from pyvale.vfm.objectivefuncvector import VectorFirstResultPassthrough
from pyvale.vfm.optimiserleastsquares import OptimiserLeastSquares
from pyvale.vfm.spatialparam import (
    initialise_parameterisations_from_constitutive_parameter,
)
from pyvale.vfm.spatialparamhomogeneous import (
    SpatialParameterisationHomogeneous,
)

EXPERIMENT_DATA_FILE = (
    Path(__file__).parent
    / "input"
    / "hole2d_plas"
    / "experiment_data.yaml"
)

KNOWN_PARAMETERS_FILE = (
    Path(__file__).parent
    / "gold"
    / "hole2d_plas.npz"
)

KNOWN_STRESS_FILE = (
    Path(__file__).parent
    / "gold"
    / "hole2d_plas_stress.npy"
)

PLOT_METRIC_IDENTIFIED_DIFF = False


# Compute virtual fields using fe model stress and known parameter
# values, then use those vfs for any further metric evaluation
@pytest.mark.skip(reason="known stress file hasn't yet been generated")
def test_sbvf_metric_with_vfs_locked():
    experiment_data = ExperimentData.load_from_file(EXPERIMENT_DATA_FILE)

    known_parameter_maps = dict(np.load(KNOWN_PARAMETERS_FILE))
    known_stress = np.load(KNOWN_STRESS_FILE)

    ident_config = _setup_identification_config()

    sbvf_metric = ident_config.phases[0].metrics[0]

    # Evaluate the sbvf metric and generate virtual fields
    # ahead of identification with known values of parameters
    # and stress
    metric_spatial_parameterisations = {
        name: [SpatialParameterisationHomogeneous()]
        for name in known_parameter_maps
    }

    parameter_map_size = np.array(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32
    )

    for name, spatial_parameterisations in metric_spatial_parameterisations.items():
        initialise_parameterisations_from_constitutive_parameter(
            spatial_parameterisations,
            ConstitutiveParameter(
                known_parameter_maps[name],
                ident_config.parameters[name].lower_bound,
                ident_config.parameters[name].upper_bound,
            ),
            parameter_map_size,
        )

    sbvf_metric.initialise(experiment_data)

    sbvf_metric.evaluate(
        known_stress,
        ident_config.constitutive_law,
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
    result = run_identification(experiment_data, ident_config)

    for name, param_map in result.parameter_maps.items():
        print(f"{name} = {np.nanmean(param_map):.6f}")

    # Copy the internal/external virtual work from the metric's final evaluation
    # during the identification (i.e. at the identified parameters).
    ivw_identified = sbvf_metric._internal_virtual_work.copy()
    evw_identified = sbvf_metric._external_virtual_work.copy()

    # ------------------------------------------------------------------
    # Test the performance of the metric: compare the SBVF metric evaluated with
    # the known (FE) stress against the metric at the identified parameters.
    # Both should give a similar residual vector. This only makes sense once the
    # identification has produced parameters to compare against.
    # ------------------------------------------------------------------
    print("Evaluating metric...")

    # Each SBVF corresponds to the single dof of one homogeneous parameter, in
    # the order the parameters are defined.
    sbvf_labels = tuple(name.replace("_", " ") for name in known_parameter_maps)

    if PLOT_METRIC_IDENTIFIED_DIFF:
        plot_metric_virtual_work(
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
@pytest.mark.skip(reason="known stress file hasn't yet been generated")
def test_sbvf_metric_with_vfs_free():
    experiment_data = ExperimentData.load_from_file(EXPERIMENT_DATA_FILE)

    known_parameter_maps = dict(np.load(KNOWN_PARAMETERS_FILE))
    known_stress = np.load(KNOWN_STRESS_FILE)

    ident_config = _setup_identification_config()

    sbvf_metric = ident_config.phases[0].metrics[0]

    # Evaluate the sbvf metric and generate virtual fields
    # ahead of identification with known values of parameters
    # and stress
    metric_spatial_parameterisations = {
        name: [SpatialParameterisationHomogeneous()]
        for name in known_parameter_maps
    }

    parameter_map_size = np.array(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32
    )

    for name, spatial_parameterisations in metric_spatial_parameterisations.items():
        initialise_parameterisations_from_constitutive_parameter(
            spatial_parameterisations,
            ConstitutiveParameter(
                known_parameter_maps[name],
                ident_config.parameters[name].lower_bound,
                ident_config.parameters[name].upper_bound,
            ),
            parameter_map_size,
        )

    sbvf_metric.initialise(experiment_data)

    sbvf_metric.evaluate(
        known_stress,
        ident_config.constitutive_law,
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
    result = run_identification(experiment_data, ident_config)

    for name, param_map in result.parameter_maps.items():
        print(f"{name} = {np.nanmean(param_map):.6f}")

    # Copy the internal/external virtual work from the metric's final evaluation
    # during the identification (i.e. at the identified parameters).
    ivw_identified = sbvf_metric._internal_virtual_work.copy()
    evw_identified = sbvf_metric._external_virtual_work.copy()

    # ------------------------------------------------------------------
    # Test the performance of the metric: compare the SBVF metric evaluated with
    # the known (FE) stress against the metric at the identified parameters.
    # Both should give a similar residual vector. This only makes sense once the
    # identification has produced parameters to compare against.
    # ------------------------------------------------------------------
    print("Evaluating metric...")

    # Each SBVF corresponds to the single dof of one homogeneous parameter, in
    # the order the parameters are defined.
    sbvf_labels = tuple(name.replace("_", " ") for name in known_parameter_maps)

    if PLOT_METRIC_IDENTIFIED_DIFF:
        plot_metric_virtual_work(
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


def _setup_identification_config() -> IdentificationConfig:
    experiment_data = ExperimentData.load_from_file(EXPERIMENT_DATA_FILE)

    parameter_map_size = np.array(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32
    )

    constitutive_law = IsotropicVonMisesElastoplasticity(HardeningLinear())

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

    metric = MetricSBVF(np.array([15, 15]))

    phases = [
        IdentificationPhase(
            {
                "elastic_modulus": [SpatialParameterisationHomogeneous()],
                "poissons_ratio": [SpatialParameterisationHomogeneous()],
                "yield_strength": [SpatialParameterisationHomogeneous()],
                "hardening_modulus": [SpatialParameterisationHomogeneous()],
            },
            [metric],
            VectorFirstResultPassthrough(),
            OptimiserLeastSquares(),
        )
    ]

    return IdentificationConfig(constitutive_law, parameters, phases)
