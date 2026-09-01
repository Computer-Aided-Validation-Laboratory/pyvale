import numpy as np
import numpy.testing as npt

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.fivephaseworkflow import (
    active_dof_summary,
    basis_amplitudes_from_snapshot,
    fixed_geometry_state_from_snapshot,
    make_phase_3_parameterisations,
    make_phase_4_parameterisations,
    make_phase_5_parameterisations,
    selected_phase_2_snapshot,
    snapshot_active_dof_summary,
)
from pyvale.vfm.identificationconfig import IdentificationPhase
from pyvale.vfm.identificationresult import PhaseResult, SolveResult, snapshot_phase
from pyvale.vfm.modelorder import (
    basis_count_from_stage,
    select_noise_resolved_basis_count,
)
from pyvale.vfm.spatialparam import PhaseSpatialState
from pyvale.vfm.spatialparambasisfuncs import (
    BasisFunctionKernelBivariateSPD,
    SpatialParameterisationBasisFunction,
)
from pyvale.vfm.spatialparamhomogeneous import SpatialParameterisationHomogeneous


def _phase_2_snapshot(basis_count: int = 2):
    x, y = np.meshgrid(np.linspace(0.0, 2.0, 3), np.linspace(0.0, 2.0, 3))
    kernels = [
        BasisFunctionKernelBivariateSPD(
            DegreeOfFreedom(0.5 + index, 0.0, 2.0),
            DegreeOfFreedom(0.75, 0.0, 2.0),
            DegreeOfFreedom(-0.2, -2.0, 2.0),
            DegreeOfFreedom(0.1, -2.0, 2.0),
            DegreeOfFreedom(0.3, -2.0, 2.0),
            0.25,
        )
        for index in range(basis_count)
    ]
    basis = SpatialParameterisationBasisFunction(
        x=x,
        y=y,
        kernels=kernels,
        heights=[
            DegreeOfFreedom(value, -1800.0, 1800.0)
            for value in (-40.0, 25.0)[:basis_count]
        ],
        kernel_type="bivariate_spd",
    )
    return snapshot_phase({
        "yield_strength": [
            SpatialParameterisationHomogeneous(
                DegreeOfFreedom(520.0, 200.0, 2000.0)
            ),
            basis,
        ]
    }), x, y


def _phase(parameterisations):
    return IdentificationPhase(
        spatial_parameterisations=parameterisations,
        metrics=[],
        objective_function=None,
        optimiser=None,
    )


def test_fixed_geometry_phase_freezing_and_releasing() -> None:
    snapshot, x, y = _phase_2_snapshot()
    geometry = fixed_geometry_state_from_snapshot(snapshot)

    phase_3 = _phase(make_phase_3_parameterisations(
        3900.0, (500.0, 10_000.0)
    ))
    assert [(row["parameter"], row["role"]) for row in active_dof_summary(phase_3)] == [
        ("hardening_modulus", "homogeneous")
    ]
    # The complete incoming yield map is represented by a fixed Known component.
    assert phase_3.spatial_parameterisations["yield_strength"][0].get_num_degrees_of_freedom() == 0

    phase_4 = _phase(make_phase_4_parameterisations(
        geometry, x, y, hardening_amplitude_bound=9500.0
    ))
    phase_4_rows = active_dof_summary(phase_4)
    assert [(row["parameter"], row["role"]) for row in phase_4_rows] == [
        ("hardening_modulus", "amplitude"),
        ("hardening_modulus", "amplitude"),
    ]
    hardening_basis = phase_4.spatial_parameterisations["hardening_modulus"][1]
    assert hardening_basis.support.get_num_degrees_of_freedom() == 0
    assert tuple(hardening_basis.kernels) != ()

    phase_5 = _phase(make_phase_5_parameterisations(
        geometry,
        x,
        y,
        hardening_homogeneous=4100.0,
        hardening_amplitudes=(100.0, -75.0),
        yield_bounds=(200.0, 2000.0),
        hardening_bounds=(500.0, 10_000.0),
    ))
    yield_basis = phase_5.spatial_parameterisations["yield_strength"][1]
    h_basis = phase_5.spatial_parameterisations["hardening_modulus"][1]
    assert yield_basis.support is h_basis.support
    assert yield_basis.support.get_num_degrees_of_freedom() == 0
    assert [(row["parameter"], row["role"]) for row in active_dof_summary(phase_5)] == [
        ("yield_strength", "homogeneous"),
        ("yield_strength", "amplitude"),
        ("yield_strength", "amplitude"),
        ("hardening_modulus", "homogeneous"),
        ("hardening_modulus", "amplitude"),
        ("hardening_modulus", "amplitude"),
    ]


def test_phase_4_to_phase_5_transfers_amplitudes_and_maps() -> None:
    snapshot, x, y = _phase_2_snapshot()
    geometry = fixed_geometry_state_from_snapshot(snapshot)
    phase_5_parameterisations = make_phase_5_parameterisations(
        geometry,
        x,
        y,
        hardening_homogeneous=4000.0,
        hardening_amplitudes=(125.0, -50.0),
        yield_bounds=(200.0, 2000.0),
        hardening_bounds=(500.0, 10_000.0),
    )
    state = PhaseSpatialState(phase_5_parameterisations)
    shape = np.asarray(x.shape, dtype=np.uint32)
    expected_yield = sum(
        item.to_map(shape)
        for item in phase_5_parameterisations["yield_strength"]
    )
    expected_hardening = sum(
        item.to_map(shape)
        for item in phase_5_parameterisations["hardening_modulus"]
    )
    parameters = {
        "elastic_modulus": ConstitutiveParameter(210_000.0, 150_000.0, 250_000.0, shape),
        "poissons_ratio": ConstitutiveParameter(0.3, 0.2, 0.4, shape),
        "yield_strength": ConstitutiveParameter(expected_yield, 200.0, 2000.0),
        "hardening_modulus": ConstitutiveParameter(expected_hardening, 500.0, 10_000.0),
    }

    state.initialise_from_constitutive_parameters(parameters, shape)

    actual = state.evaluate_parameter_maps(shape)
    npt.assert_allclose(actual["yield_strength"], expected_yield)
    npt.assert_allclose(actual["hardening_modulus"], expected_hardening)
    persisted = snapshot_phase(state.spatial_parameterisations)
    assert basis_amplitudes_from_snapshot(
        persisted, "hardening_modulus"
    ) == (125.0, -50.0)


def test_selected_snapshot_uses_requested_persisted_model_order() -> None:
    bf1, _, _ = _phase_2_snapshot(1)
    bf2, _, _ = _phase_2_snapshot(2)
    result = PhaseResult(solve_results=[
        SolveResult(final_snapshot=bf1),
        SolveResult(final_snapshot=bf2),
    ])

    selected = selected_phase_2_snapshot(result, 1)

    assert fixed_geometry_state_from_snapshot(selected).basis_count == 1


def test_phase_2_snapshot_labels_geometry_and_amplitude_dofs() -> None:
    snapshot, _, _ = _phase_2_snapshot(2)

    rows = snapshot_active_dof_summary(snapshot, include_geometry=True)

    roles = [row["role"] for row in rows]
    assert roles.count("geometry") == 10
    assert roles.count("homogeneous") == 1
    assert roles.count("amplitude") == 2
    assert [
        row["name"] for row in rows if row["role"] == "geometry"
    ][:5] == [
        "centre_x", "centre_y", "log_covariance_11",
        "log_covariance_12", "log_covariance_22",
    ]


def test_noise_resolved_selector_can_recover_after_a_failed_transition() -> None:
    rows = [
        {"transition": "Phase0→BF1", "child_stage": "BF1", "pass": True,
         "parent_j": 10.0, "child_j": 8.0, "q95_absolute_noise_change": 0.5},
        {"transition": "BF1→BF2", "child_stage": "BF2", "pass": False,
         "parent_j": 8.0, "child_j": 7.8, "q95_absolute_noise_change": 0.5},
        {"transition": "BF2→BF3", "child_stage": "BF3", "pass": True,
         "parent_j": 7.8, "child_j": 7.0, "q95_absolute_noise_change": 0.5},
    ]

    selection = select_noise_resolved_basis_count(rows)

    assert selection["first_fail_selected"] == "BF1"
    assert selection["cumulative_selected"] == "BF3"
    assert basis_count_from_stage(selection["cumulative_selected"]) == 3


def test_noise_resolved_selector_does_not_hard_code_bf5() -> None:
    rows = [
        {"transition": f"BF{index - 1}→BF{index}", "child_stage": f"BF{index}",
         "pass": True, "parent_j": 10.0-index, "child_j": 9.0-index,
         "q95_absolute_noise_change": 0.1}
        for index in range(1, 8)
    ]

    selection = select_noise_resolved_basis_count(rows)

    assert selection["cumulative_selected"] == "BF7"
