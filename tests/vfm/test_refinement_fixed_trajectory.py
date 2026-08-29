from pyvale.vfm.refinement import EquilibriumGapBasisGrowthRefinement


def test_fixed_basis_trajectory_is_explicitly_configurable():
    policy = EquilibriumGapBasisGrowthRefinement(
        target=object(),
        max_basis_functions=7,
        relative_improvement_threshold=0.0,
        fixed_basis_trajectory=True,
    )
    assert policy.fixed_basis_trajectory is True
