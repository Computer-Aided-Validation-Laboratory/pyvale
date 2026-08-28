from __future__ import annotations

import numpy as np
import pytest

from pyvale.vfm.constlaw import EIdentificationType, IConstitutiveLaw
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import (
    BoundaryConditions,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.metricsbvf import (
    StressSensitivity,
    calculate_local_parameter_stress_sensitivity,
    calculate_parameter_stress_sensitivities,
)
from pyvale.vfm.metricequilibriumgap import EquilibriumGapMetric
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.identification import PhaseRuntime
from pyvale.vfm.objectivefunccombinedfreegi import (
    CombinedForceAndEquilibriumGapObjective,
)
from pyvale.vfm.roi import RoiDefinition, RoiShape, VfmRegionOfInterest
from pyvale.vfm.refinement import _basis_from_correction_map
from pyvale.vfm.slicewise_utils import SliceConfig
from pyvale.vfm.spatialweighting import (
    SensitivitySpatialWeightingConfig,
    resolve_sensitivity_spatial_weights,
)
from pyvale.vfm.spatialparamhomogeneous import SpatialParameterisationHomogeneous
from pyvale.vfm.spatialparambasisfuncs import (
    BasisFunctionKernelBivariateSPD,
    SpatialParameterisationBasisFunction,
)


class _LinearMapLaw(IConstitutiveLaw):
    def get_identification_type(self) -> EIdentificationType:
        return EIdentificationType.Nonlinear

    def get_required_parameters(self) -> list[str]:
        return ["first", "second"]

    def calculate_stress(
        self,
        strain: np.ndarray,
        constitutive_parameter_maps: dict[str, np.ndarray],
    ) -> np.ndarray:
        stress = np.zeros_like(strain)
        stress[:, 0] = (
            strain[:, 0] * constitutive_parameter_maps["first"]
            + 2.0 * constitutive_parameter_maps["second"]
        )
        return stress


def test_force_residual_stress_adjoint_matches_finite_difference() -> None:
    experiment_data = _rectangle_experiment_data()
    metric = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis="x", num_slices=5)
    )
    metric.initialise(experiment_data)
    rng = np.random.default_rng(7)
    stress = rng.normal(size=experiment_data.strain.shape)
    direction = rng.normal(size=stress.shape)
    residual = metric.evaluate_force_recon_error(
        stress, experiment_data
    ).metric_result.additional_fields["normalised_residual"]
    cotangent = rng.normal(size=residual.shape)
    stress_cotangent = metric.normalised_residual_stress_adjoint(
        cotangent,
        experiment_data,
    )
    predicted = float(np.sum(stress_cotangent * direction))
    step = 1.0e-6
    plus = metric.evaluate_force_recon_error(
        stress + step * direction, experiment_data
    ).metric_result.additional_fields["normalised_residual"]
    minus = metric.evaluate_force_recon_error(
        stress - step * direction, experiment_data
    ).metric_result.additional_fields["normalised_residual"]
    observed = float(np.nansum(cotangent * (plus - minus)) / (2.0 * step))

    assert predicted == pytest.approx(observed, rel=1.0e-7)


def test_correction_growth_fits_continuous_signed_spd_feature() -> None:
    experiment_data = _rectangle_experiment_data()
    x = experiment_data.specimen_geometry.x
    y = experiment_data.specimen_geometry.y
    correction = -np.exp(-0.5 * (((x - 7.0) / 2.0) ** 2 + (y - 4.0) ** 2))
    basis = SpatialParameterisationBasisFunction(
        x,
        y,
        kernel_type="bivariate_spd",
    )
    parameter = ConstitutiveParameter(np.full(x.shape, 400.0), 200.0, 800.0)

    proposal = _basis_from_correction_map(
        correction,
        basis,
        experiment_data,
        parameter,
        height_fraction=0.05,
        smoothing_points=1,
        minimum_separation_points=0.0,
        feature_fraction=0.2,
    )

    assert proposal is not None
    kernel, height, diagnostics = proposal
    assert isinstance(kernel, BasisFunctionKernelBivariateSPD)
    assert height.value < 0.0
    assert np.all(np.linalg.eigvalsh(kernel.covariance()) > 0.0)
    assert diagnostics["policy"] == "sensitivity_correction"


def _rectangle_experiment_data() -> ExperimentData:
    rows, columns = 9, 11
    forces = np.array([10.0, 20.0, 30.0])
    x, y = np.meshgrid(
        np.arange(columns, dtype=np.float64),
        np.arange(rows, dtype=np.float64),
    )
    roi = VfmRegionOfInterest.from_definition(
        RoiDefinition(
            shapes=(
                RoiShape(
                    shape_type="rectangle",
                    index=0,
                    is_cutting=False,
                    rectangle=(0.0, 0.0, float(columns - 1), float(rows - 1)),
                ),
            ),
        )
    )
    geometry = SpecimenGeometry(
        x,
        y,
        np.ones((rows, columns), dtype=np.float64),
        1.0,
        roi,
    )
    boundary_conditions = BoundaryConditions(
        EdgeConditions(
            min_x_edge=Edge(EEdgeCondition.Fixed, EEdgeCondition.Free),
            max_x_edge=Edge(EEdgeCondition.Traction, EEdgeCondition.Free),
            min_y_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Free),
            max_y_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Free),
        ),
        np.column_stack((forces, np.zeros_like(forces))),
    )
    strain = np.zeros((forces.size, 3, rows, columns), dtype=np.float64)
    return ExperimentData(
        strain,
        geometry,
        boundary_conditions,
        np.arange(forces.size, dtype=np.float64),
    )


def test_parameter_stress_sensitivities_are_labelled_and_do_not_mutate_maps() -> None:
    strain = np.zeros((3, 3, 2, 2), dtype=np.float64)
    strain[:, 0] = np.arange(1.0, 4.0)[:, np.newaxis, np.newaxis]
    parameter_maps = {
        "first": np.full((2, 2), 10.0),
        "second": np.full((2, 2), 4.0),
    }
    original_maps = {name: values.copy() for name, values in parameter_maps.items()}
    law = _LinearMapLaw()
    reference = law.calculate_stress(strain, parameter_maps)

    sensitivities = calculate_parameter_stress_sensitivities(
        strain,
        reference,
        law,
        parameter_maps,
        ["second", "first"],
        perturbation_factor=0.1,
    )

    assert tuple(sensitivities) == ("second", "first")
    np.testing.assert_allclose(sensitivities["second"].total[:, 0], 0.8)

    local = calculate_local_parameter_stress_sensitivity(
        strain,
        reference,
        law,
        parameter_maps,
        "second",
        perturbation_factor=0.1,
    )
    np.testing.assert_allclose(local[:, 0], 2.0)
    np.testing.assert_allclose(
        sensitivities["first"].total[:, 0],
        np.broadcast_to(
            np.arange(1.0, 4.0)[:, np.newaxis, np.newaxis],
            (3, 2, 2),
        ),
    )
    np.testing.assert_allclose(sensitivities["first"].incremental[0], 0.0)
    np.testing.assert_allclose(sensitivities["first"].incremental[1:, 0], 1.0)
    for name in parameter_maps:
        np.testing.assert_array_equal(parameter_maps[name], original_maps[name])


def test_resolved_weights_are_normalised_and_emphasise_sensitive_support() -> None:
    experiment_data = _rectangle_experiment_data()
    rows, columns = experiment_data.specimen_geometry.x.shape
    force_history = experiment_data.boundary_conditions.force[:, 0]

    localised = np.zeros((force_history.size, 3, rows, columns), dtype=np.float64)
    localised[:, 0, :, columns // 2 :] = force_history[:, np.newaxis, np.newaxis]
    localised[:, 1, rows // 2 :, :] = 0.5 * force_history[:, np.newaxis, np.newaxis]
    diffuse = np.zeros_like(localised)
    diffuse[:, 0] = (
        force_history[:, np.newaxis, np.newaxis]
        * np.linspace(0.0, 0.2, columns)[np.newaxis, np.newaxis, :]
    )
    sensitivities = {
        "yield_strength": StressSensitivity(localised, np.zeros_like(localised)),
        "hardening_modulus": StressSensitivity(diffuse, np.zeros_like(diffuse)),
    }

    egi_metric = EquilibriumGapMetric(window_size=(3, 3))
    force_metric = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis="x", num_slices=5)
    )
    egi_metric.initialise(experiment_data)
    force_metric.initialise(experiment_data)
    resolved = resolve_sensitivity_spatial_weights(
        sensitivities,
        [egi_metric],
        force_metric,
        experiment_data,
    )

    assert resolved.parameter_names == ("yield_strength", "hardening_modulus")
    egi_weights = resolved.equilibrium_gap_weights[0]
    valid_egi_weights = egi_weights[np.isfinite(egi_weights)]
    assert np.mean(valid_egi_weights) == pytest.approx(1.0)
    assert np.max(valid_egi_weights) > np.min(valid_egi_weights)
    assert np.sum(resolved.force_weights) == pytest.approx(1.0)
    assert resolved.force_weights[-1] > resolved.force_weights[0]


def test_zero_sensitivity_falls_back_to_uniform_egi_and_slice_width_weights() -> None:
    experiment_data = _rectangle_experiment_data()
    shape = experiment_data.strain.shape
    zero = np.zeros(shape, dtype=np.float64)
    sensitivities = {
        "yield_strength": StressSensitivity(zero, zero.copy()),
    }
    egi_metric = EquilibriumGapMetric(window_size=(3, 3))
    force_metric = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis="x", num_slices=5)
    )
    egi_metric.initialise(experiment_data)
    force_metric.initialise(experiment_data)

    resolved = resolve_sensitivity_spatial_weights(
        sensitivities,
        [egi_metric],
        force_metric,
        experiment_data,
    )

    egi_weights = resolved.equilibrium_gap_weights[0]
    np.testing.assert_allclose(egi_weights[np.isfinite(egi_weights)], 1.0)
    assert force_metric.slice_partition is not None
    expected_force_weights = force_metric.slice_partition.widths
    expected_force_weights = expected_force_weights / np.sum(expected_force_weights)
    np.testing.assert_allclose(resolved.force_weights, expected_force_weights)


def test_phase_runtime_resolves_weights_once_and_prepare_does_not_change_them() -> None:
    experiment_data = _rectangle_experiment_data()
    shape = experiment_data.specimen_geometry.x.shape
    experiment_data.strain[:, 0] = np.arange(1.0, 4.0)[:, None, None] * 0.01
    parameters = {
        "first": ConstitutiveParameter(10.0, 1.0, 20.0, np.asarray(shape)),
        "second": ConstitutiveParameter(4.0, 1.0, 10.0, np.asarray(shape)),
    }
    egi_metric = EquilibriumGapMetric(window_size=(3, 3))
    force_metric = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis="x", num_slices=5)
    )
    objective = CombinedForceAndEquilibriumGapObjective(
        egi_window_weights=(1.0,),
        spatial_weighting=SensitivitySpatialWeightingConfig(),
    )
    runtime = PhaseRuntime(
        spatial_parameterisations={
            "first": [SpatialParameterisationHomogeneous()],
            "second": [SpatialParameterisationHomogeneous()],
        },
        metrics=[force_metric, egi_metric],
        objective_function=objective,
    )
    runtime.prepare(experiment_data)
    runtime.resolve_spatial_weighting(_LinearMapLaw(), parameters, experiment_data)
    assert objective.resolved_spatial_weights is not None
    before_egi = tuple(
        weights.copy()
        for weights in objective.resolved_spatial_weights.equilibrium_gap_weights
    )
    before_force = objective.resolved_spatial_weights.force_weights.copy()

    runtime.prepare(experiment_data)

    assert objective.resolved_spatial_weights is not None
    for before, after in zip(
        before_egi,
        objective.resolved_spatial_weights.equilibrium_gap_weights,
        strict=True,
    ):
        np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(
        objective.resolved_spatial_weights.force_weights,
        before_force,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"perturbation_factor": 0.0}, "perturbation_factor"),
        ({"weight_floor": 0.0}, "weight_floor"),
        ({"scaling_percentile": 101.0}, "scaling_percentile"),
    ),
)
def test_weighting_config_rejects_invalid_values(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SensitivitySpatialWeightingConfig(**kwargs)
