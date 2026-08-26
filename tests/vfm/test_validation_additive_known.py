import numpy as np

from pyvale.vfm import (
    ConstitutiveParameter,
    CombinedForceAndEquilibriumGapObjective,
    CombinedObjectiveBaseline,
    DegreeOfFreedom,
    EquilibriumGapBasisGrowthRefinement,
    HardeningLinear,
    IdentificationConfig,
    IdentificationPhase,
    IsotropicVonMisesElastoplasticity,
    OptimiserPatternSearch,
    ScalarFirstResultPassthrough,
    SpatialParameterisationBasisFunction,
    SpatialParameterisationHomogeneous,
    SpatialParameterisationKnown,
)
from pyvale.vfm.spatialparambasisfuncs import BasisFunctionKernelUnivariate
from pyvale.vfm.validation import _collect_identification_config_errors


def test_known_baseline_may_have_active_additive_basis_increment() -> None:
    x, y = np.meshgrid(np.arange(2, dtype=float), np.arange(2, dtype=float))
    basis = SpatialParameterisationBasisFunction(x, y)
    basis.kernels.append(BasisFunctionKernelUnivariate(
        0.5, 0.5, DegreeOfFreedom(1.0, 0.1, 10.0, scaling="log")
    ))
    basis.heights.append(DegreeOfFreedom(20.0, -100.0, 100.0))
    shape = np.asarray((2, 2), dtype=np.uint32)
    parameters = {
        "elastic_modulus": ConstitutiveParameter(190_000.0, 150_000.0, 250_000.0, shape),
        "poissons_ratio": ConstitutiveParameter(0.28, 0.2, 0.4, shape),
        "yield_strength": ConstitutiveParameter(360.0, 200.0, 700.0, shape),
        "hardening_modulus": ConstitutiveParameter(3_500.0, 500.0, 10_000.0, shape),
    }
    phase = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [SpatialParameterisationKnown(), basis],
            "hardening_modulus": [SpatialParameterisationKnown()],
        },
        metrics=[object()], objective_function=ScalarFirstResultPassthrough(),
        optimiser=OptimiserPatternSearch(max_evaluations=2, max_iterations=1),
    )
    config = IdentificationConfig(
        IsotropicVonMisesElastoplasticity(HardeningLinear()), parameters, [phase]
    )

    assert _collect_identification_config_errors(config) == []


def test_phase_rejects_multiple_basis_parameterisations_for_one_parameter() -> None:
    x, y = np.meshgrid(np.arange(2, dtype=float), np.arange(2, dtype=float))
    shape = np.asarray((2, 2), dtype=np.uint32)
    parameters = {
        "elastic_modulus": ConstitutiveParameter(190_000.0, 150_000.0, 250_000.0, shape),
        "poissons_ratio": ConstitutiveParameter(0.28, 0.2, 0.4, shape),
        "yield_strength": ConstitutiveParameter(360.0, 200.0, 700.0, shape),
        "hardening_modulus": ConstitutiveParameter(3_500.0, 500.0, 10_000.0, shape),
    }
    phase = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [
                SpatialParameterisationBasisFunction(x, y),
                SpatialParameterisationBasisFunction(x, y),
            ],
            "hardening_modulus": [SpatialParameterisationHomogeneous()],
        },
        metrics=[object()],
        objective_function=ScalarFirstResultPassthrough(),
        optimiser=OptimiserPatternSearch(max_evaluations=2, max_iterations=1),
    )
    config = IdentificationConfig(
        IsotropicVonMisesElastoplasticity(HardeningLinear()), parameters, [phase]
    )

    assert any(
        "yield_strength" in error
        and "at most one SpatialParameterisationBasisFunction" in error
        for error in _collect_identification_config_errors(config)
    )


def test_prior_phase_baseline_must_reference_an_earlier_phase() -> None:
    shape = np.asarray((2, 2), dtype=np.uint32)
    parameters = {
        "elastic_modulus": ConstitutiveParameter(190_000.0, 150_000.0, 250_000.0, shape),
        "poissons_ratio": ConstitutiveParameter(0.28, 0.2, 0.4, shape),
        "yield_strength": ConstitutiveParameter(360.0, 200.0, 700.0, shape),
        "hardening_modulus": ConstitutiveParameter(3_500.0, 500.0, 10_000.0, shape),
    }
    spatial_parameterisations = {
        "elastic_modulus": [SpatialParameterisationKnown()],
        "poissons_ratio": [SpatialParameterisationKnown()],
        "yield_strength": [SpatialParameterisationKnown(), SpatialParameterisationHomogeneous()],
        "hardening_modulus": [SpatialParameterisationKnown()],
    }
    phase_0 = IdentificationPhase(
        spatial_parameterisations=spatial_parameterisations,
        metrics=[object()],
        objective_function=CombinedForceAndEquilibriumGapObjective(
            egi_window_weights=(1.0,),
        ),
        optimiser=OptimiserPatternSearch(max_evaluations=2, max_iterations=1),
    )
    phase_1 = IdentificationPhase(
        spatial_parameterisations=spatial_parameterisations,
        metrics=[object()],
        objective_function=CombinedForceAndEquilibriumGapObjective(
            egi_window_weights=(1.0,),
            baseline=CombinedObjectiveBaseline.prior_phase(1),
        ),
        optimiser=OptimiserPatternSearch(max_evaluations=2, max_iterations=1),
    )
    config = IdentificationConfig(
        IsotropicVonMisesElastoplasticity(HardeningLinear()),
        parameters,
        [phase_0, phase_1],
    )

    errors = _collect_identification_config_errors(config)

    assert errors == [
        "phase 1: prior-phase baseline must reference an earlier phase, "
        "got phase_index 1"
    ]


def test_egi_basis_growth_requires_an_egi_metric() -> None:
    shape = np.asarray((2, 2), dtype=np.uint32)
    parameters = {
        "elastic_modulus": ConstitutiveParameter(190_000.0, 150_000.0, 250_000.0, shape),
        "poissons_ratio": ConstitutiveParameter(0.28, 0.2, 0.4, shape),
        "yield_strength": ConstitutiveParameter(360.0, 200.0, 700.0, shape),
        "hardening_modulus": ConstitutiveParameter(3_500.0, 500.0, 10_000.0, shape),
    }
    x, y = np.meshgrid(np.arange(2, dtype=float), np.arange(2, dtype=float))
    basis = SpatialParameterisationBasisFunction(x, y)
    phase = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [
                SpatialParameterisationHomogeneous(),
                basis,
            ],
            "hardening_modulus": [SpatialParameterisationKnown()],
        },
        metrics=[object()],
        objective_function=CombinedForceAndEquilibriumGapObjective(
            egi_window_weights=(1.0,),
        ),
        optimiser=OptimiserPatternSearch(max_evaluations=2, max_iterations=1),
        refinement_policy=EquilibriumGapBasisGrowthRefinement(
            target=basis,
        ),
    )
    config = IdentificationConfig(
        IsotropicVonMisesElastoplasticity(HardeningLinear()), parameters, [phase]
    )

    assert _collect_identification_config_errors(config) == [
        "phase 0: EGI basis growth requires at least one EquilibriumGapMetric"
    ]
