import numpy as np
import pytest

from pyvale.vfm.materialprojection import (
    bound_aware_sensitivity,
    build_material_projection_bases,
    central_difference_sensitivity,
)


def test_rank_revealing_projection_and_yield_unique_residualisation():
    sensitivity = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    bases = build_material_projection_bases(
        sensitivity, parameter_groups=("yield", "hardening", "yield")
    )
    assert bases.full.rank == 2
    assert bases.yield_basis.rank == 2
    assert bases.hardening_basis.rank == 1
    assert bases.yield_unique.rank == 1
    assert bases.yield_hardening_max_correlation == pytest.approx(1.0)


def test_diagonal_whitening_changes_basis_and_records_diagnostics():
    bases = build_material_projection_bases(
        [[1.0], [1.0]],
        parameter_groups=("yield",),
        observation_scale=[1.0, 2.0],
    )
    expected = np.array([1.0, 0.5]) / np.sqrt(1.25)
    assert np.abs(bases.full.basis[:, 0]) == pytest.approx(expected)
    assert bases.diagnostics()["full"]["rank"] == 1


def test_central_difference_restores_reference_and_reports_progress():
    calls = []
    progress = []

    def residual(dofs):
        calls.append(np.asarray(dofs).copy())
        return np.array([2.0 * dofs[0] + dofs[1], dofs[0] - 3.0 * dofs[1]])

    derivative = central_difference_sensitivity(
        [0.2, -0.1], residual, step=1e-5,
        progress_callback=lambda complete, total: progress.append((complete, total)),
    )
    np.testing.assert_allclose(derivative, [[2.0, 1.0], [1.0, -3.0]])
    np.testing.assert_allclose(calls[-1], [0.2, -0.1])
    assert progress == [(1, 2), (2, 2)]


def test_bound_aware_sensitivity_uses_one_sided_steps_at_bounds():
    reference = np.array([0.0, 0.5, 1.0])
    calls = []

    def residual(dofs):
        calls.append(np.asarray(dofs).copy())
        return np.array([
            dofs[0] + 2.0 * dofs[1],
            dofs[1] ** 2 + 3.0 * dofs[2],
        ])

    result = bound_aware_sensitivity(reference, residual, step=1.0e-4)

    assert result.schemes == ("forward", "central", "backward")
    np.testing.assert_allclose(
        result.matrix,
        [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]],
        atol=2.0e-4,
    )
    np.testing.assert_allclose(calls[-1], reference)


def test_bound_aware_sensitivity_restores_reference_after_failure():
    reference = np.array([0.5])
    calls = []

    def residual(dofs):
        values = np.asarray(dofs)
        calls.append(values.copy())
        if values[0] > reference[0]:
            return np.array([np.nan])
        return values.copy()

    with pytest.raises(ValueError, match="finite"):
        bound_aware_sensitivity(reference, residual)

    np.testing.assert_allclose(calls[-1], reference)


def test_bound_aware_sensitivity_uses_the_roomier_side_near_a_bound():
    reference = np.array([1.0 - 1.0e-8])

    result = bound_aware_sensitivity(
        reference,
        lambda dofs: np.asarray(dofs) ** 2,
        step=1.0e-3,
    )

    assert result.schemes == ("backward",)
    assert result.step_sizes == pytest.approx((1.0e-3,))
    assert result.matrix[0, 0] == pytest.approx(2.0, abs=1.1e-3)


def test_inactive_hardening_has_no_hardening_or_unique_penalty_failure():
    bases = build_material_projection_bases(
        [[1.0], [0.0]], parameter_groups=("yield",)
    )
    assert bases.hardening_basis is None
    assert bases.yield_unique.rank == 1
