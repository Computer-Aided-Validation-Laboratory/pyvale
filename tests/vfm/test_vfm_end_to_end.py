from __future__ import annotations

import numpy as np
import numpy.testing as npt

from pyvale.examples.vfm.synthetic_rectangular_tensile import (
    SyntheticTensileMaterial,
    build_synthetic_identification_parameters,
    build_synthetic_tensile_case,
)
from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.hardening import HardeningLinear
from pyvale.vfm.identification import run_identification
from pyvale.vfm.identificationconfig import (
    IdentificationConfig,
    IdentificationPhase,
)
from pyvale.vfm.metricsbvf import MetricSBVF
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.objectivefuncvector import (
    VectorFirstResultPassthrough,
    VectorWeightedObjective,
)
from pyvale.vfm.optimiserleastsquares import OptimiserLeastSquares
from pyvale.vfm.slicewise_utils import SliceConfig
from pyvale.vfm.spatialparamhomogeneous import (
    SpatialParameterisationHomogeneous,
)
from pyvale.vfm.spatialparamknown import SpatialParameterisationKnown
from pyvale.vfm.spatialparamslicewise import (
    SliceWiseSpatialParameterisation,
    SupportSlice,
)


def test_end_to_end_homogeneous_sbvf_identification() -> None:
    case = build_synthetic_tensile_case(
        num_grid_columns=12,
        length=12.0,
        material=SyntheticTensileMaterial(yield_strengths=(250.0,)),
    )
    map_size = np.asarray(
        case.experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32,
    )
    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            180_000.0, 150_000.0, 260_000.0, map_size
        ),
        "poissons_ratio": ConstitutiveParameter(
            0.25, 0.15, 0.45, map_size
        ),
        "yield_strength": ConstitutiveParameter(
            220.0, 150.0, 350.0, map_size
        ),
        "hardening_modulus": ConstitutiveParameter(
            5_000.0, 1_000.0, 15_000.0, map_size
        ),
    }
    phase = IdentificationPhase(
        spatial_parameterisations={
            name: [SpatialParameterisationHomogeneous()]
            for name in parameters
        },
        metrics=[MetricSBVF(np.array([2, 3], dtype=np.uint32))],
        objective_function=VectorFirstResultPassthrough(),
        optimiser=OptimiserLeastSquares(max_evaluations=100),
        optimisation_newton_tolerance=1.0e-8,
    )

    result = run_identification(
        case.experiment_data,
        IdentificationConfig(
            constitutive_law=IsotropicVonMisesElastoplasticity(
                HardeningLinear(),
                error_tolerance=1.0e-10,
            ),
            parameters=parameters,
            phases=[phase],
        ),
    )

    for name, expected_map in case.known_parameter_maps.items():
        npt.assert_allclose(
            result.parameter_maps[name],
            expected_map,
            rtol=5.0e-6,
            atol=1.0e-6,
        )

    solve_result = result.history.phases[0].solve_results[-1]
    assert solve_result.success is True
    assert solve_result.final_objective is not None
    assert solve_result.final_objective["residual_norm"] < 1.0e-2


def test_end_to_end_homogeneous_then_slicewise_identification() -> None:
    case = build_synthetic_tensile_case()
    map_size = np.asarray(
        case.experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32,
    )
    parameters = build_synthetic_identification_parameters(
        map_size,
        material=case.material,
        initial_yield_strength=250.0,
    )

    homogeneous_phase = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [SpatialParameterisationHomogeneous()],
            "hardening_modulus": [SpatialParameterisationHomogeneous()],
        },
        metrics=[MetricSBVF(np.array([2, 3], dtype=np.uint32))],
        objective_function=VectorFirstResultPassthrough(),
        optimiser=OptimiserLeastSquares(max_evaluations=100),
        optimisation_newton_tolerance=1.0e-8,
    )
    support = SupportSlice(
        slice_config=SliceConfig(axis="x", num_slices=4),
    )
    slicewise_phase = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [
                SliceWiseSpatialParameterisation(support=support)
            ],
            "hardening_modulus": [SpatialParameterisationHomogeneous()],
        },
        metrics=[SliceWiseForceReconstructionMetric(support=support)],
        objective_function=VectorWeightedObjective(),
        optimiser=OptimiserLeastSquares(max_evaluations=100),
        optimisation_newton_tolerance=1.0e-8,
    )

    result = run_identification(
        case.experiment_data,
        IdentificationConfig(
            constitutive_law=IsotropicVonMisesElastoplasticity(
                HardeningLinear(),
                error_tolerance=1.0e-10,
            ),
            parameters=parameters,
            phases=[homogeneous_phase, slicewise_phase],
        ),
    )

    assert len(result.history.phases) == 2
    phase_zero = result.history.phases[0].spatial_parameterisations
    assert phase_zero["elastic_modulus"][0].summary["kind"] == "known"
    assert phase_zero["poissons_ratio"][0].summary["kind"] == "known"
    assert phase_zero["yield_strength"][0].summary["kind"] == "homogeneous"
    assert phase_zero["hardening_modulus"][0].summary["kind"] == "homogeneous"

    phase_one = result.history.phases[1].spatial_parameterisations
    assert phase_one["yield_strength"][0].summary["kind"] == "slice_wise"
    assert phase_one["hardening_modulus"][0].summary["kind"] == "homogeneous"

    for name, expected_map in case.known_parameter_maps.items():
        npt.assert_allclose(
            result.parameter_maps[name],
            expected_map,
            rtol=1.0e-7,
            atol=1.0e-5,
        )

    phase_one_solve = result.history.phases[1].solve_results[-1]
    assert phase_one_solve.success is True
    assert phase_one_solve.final_objective is not None
    assert phase_one_solve.final_objective["residual_norm"] < 1.0e-5
