from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from pyvale.examples.vfm.synthetic_rectangular_tensile import (
    SyntheticTensileCase,
    SyntheticTensileMaterial,
    build_synthetic_tensile_case,
)
from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.hardening import HardeningLinear
from pyvale.vfm.metricsbvf import MetricSBVF
from pyvale.vfm.spatialparam import (
    initialise_parameterisations_from_constitutive_parameter,
)
from pyvale.vfm.spatialparamhomogeneous import (
    SpatialParameterisationHomogeneous,
)


@pytest.fixture(scope="module")
def homogeneous_case() -> SyntheticTensileCase:
    return build_synthetic_tensile_case(
        num_grid_columns=12,
        length=12.0,
        material=SyntheticTensileMaterial(yield_strengths=(250.0,)),
    )


def test_constitutive_law_reconstructs_analytical_uniaxial_stress(
    homogeneous_case: SyntheticTensileCase,
) -> None:
    law = IsotropicVonMisesElastoplasticity(
        HardeningLinear(),
        error_tolerance=1.0e-10,
    )

    reconstructed = law.calculate_stress(
        homogeneous_case.experiment_data.strain,
        homogeneous_case.known_parameter_maps,
    )

    npt.assert_allclose(
        reconstructed,
        homogeneous_case.known_stress,
        rtol=0.0,
        atol=1.0e-6,
    )


def test_sbvf_metric_balances_known_internal_and_external_work(
    homogeneous_case: SyntheticTensileCase,
) -> None:
    metric = MetricSBVF(np.array([2, 3], dtype=np.uint32))
    metric.initialise(homogeneous_case.experiment_data)

    result = metric.evaluate(
        homogeneous_case.known_stress,
        IsotropicVonMisesElastoplasticity(HardeningLinear()),
        _map_size(homogeneous_case),
        _known_homogeneous_parameterisations(homogeneous_case),
        homogeneous_case.experiment_data,
    )

    npt.assert_allclose(result.residual, 0.0, rtol=0.0, atol=1.0e-8)
    assert metric._internal_virtual_work is not None
    assert metric._external_virtual_work is not None
    npt.assert_allclose(
        metric._internal_virtual_work,
        metric._external_virtual_work,
        rtol=0.0,
        atol=1.0e-8,
    )


def test_sbvf_metric_can_reuse_or_recompute_virtual_fields(
    homogeneous_case: SyntheticTensileCase,
) -> None:
    metric = MetricSBVF(np.array([2, 3], dtype=np.uint32))
    metric.initialise(homogeneous_case.experiment_data)
    law = IsotropicVonMisesElastoplasticity(HardeningLinear())
    parameterisations = _known_homogeneous_parameterisations(homogeneous_case)

    metric.evaluate(
        homogeneous_case.known_stress,
        law,
        _map_size(homogeneous_case),
        parameterisations,
        homogeneous_case.experiment_data,
    )
    initial_fields = metric._sensitivity_based_virtual_fields
    assert initial_fields is not None

    metric._recompute_virtual_fields = False
    metric.evaluate(
        homogeneous_case.known_stress,
        law,
        _map_size(homogeneous_case),
        parameterisations,
        homogeneous_case.experiment_data,
    )
    assert metric._sensitivity_based_virtual_fields is initial_fields

    metric._recompute_virtual_fields = True
    metric.evaluate(
        homogeneous_case.known_stress,
        law,
        _map_size(homogeneous_case),
        parameterisations,
        homogeneous_case.experiment_data,
    )
    assert metric._sensitivity_based_virtual_fields is not initial_fields


def _known_homogeneous_parameterisations(
    case: SyntheticTensileCase,
) -> dict[str, list[SpatialParameterisationHomogeneous]]:
    bounds = {
        "elastic_modulus": (100_000.0, 300_000.0),
        "poissons_ratio": (0.2, 0.4),
        "yield_strength": (100.0, 500.0),
        "hardening_modulus": (1_000.0, 20_000.0),
    }
    parameterisations: dict[
        str, list[SpatialParameterisationHomogeneous]
    ] = {}
    for name, parameter_map in case.known_parameter_maps.items():
        lower_bound, upper_bound = bounds[name]
        parameterisations[name] = [SpatialParameterisationHomogeneous()]
        initialise_parameterisations_from_constitutive_parameter(
            parameterisations[name],
            ConstitutiveParameter(
                parameter_map,
                lower_bound,
                upper_bound,
            ),
            _map_size(case),
        )
    return parameterisations


def _map_size(case: SyntheticTensileCase) -> np.ndarray:
    return np.asarray(
        case.experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32,
    )
