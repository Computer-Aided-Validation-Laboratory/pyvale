import pyvale.vfm as vfm


REQUIRED_EXPORTS = {
    "run_identification",
    "IdentificationConfig",
    "IdentificationPhase",
    "IsotropicVonMisesElastoplasticity",
    "SpatialParameterisationKnown",
    "SpatialParameterisationHomogeneous",
    "SliceWiseSpatialParameterisation",
    "SupportSlice",
    "MetricSBVF",
    "SliceWiseForceReconstructionMetric",
    "EquilibriumGapMetric",
    "EquilibriumGapResult",
    "EquilibriumGapVirtualFieldType",
    "OptimiserLeastSquares",
    "SliceWiseIndependentLeastSquares",
    "VectorFirstResultPassthrough",
    "VectorWeightedObjective",
    "SliceMergeSplitRefinement",
}

DEFERRED_EXPORTS = {
    "SpatialParameterisationBasisFunction",
    "SupportBasis",
    "OptimiserPatternSearch",
    "ScalarFirstResultPassthrough",
    "VectorConcatenateObjective",
    "ScalarForceAndEquilibriumGapObjective",
    "CombinedForceAndEquilibriumGapObjective",
    "BasisAddRemoveRefinement",
    "EquilibriumGapBasisGrowthRefinement",
}


def test_initial_stable_public_api() -> None:
    exported = set(vfm.__all__)

    assert REQUIRED_EXPORTS <= exported
    assert exported.isdisjoint(DEFERRED_EXPORTS)
    assert all(hasattr(vfm, name) for name in exported)