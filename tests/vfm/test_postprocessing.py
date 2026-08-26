import numpy as np
import numpy.testing as npt
import pytest
from types import SimpleNamespace

from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.hardening import HardeningLinear
from pyvale.vfm.identificationresult import (
    ConfigSnapshot,
    IdentificationHistory,
    IdentificationMetadata,
    IdentificationResult,
    ObjectSnapshot,
    PhaseResult,
    SolveResult,
    snapshot_phase,
)
from pyvale.vfm.postprocessing import (
    ParameterErrorDiagnostics,
    cache_parameter_error_diagnostics,
    check_stress_against_saved,
    component_history_map,
    compute_parameter_error_diagnostics,
    evaluate_snapshot_parameter_maps,
    load_constitutive_law_from_result,
    parameter_map_summary,
    plot_phase_objective_histories,
    plot_solve_parameter_maps,
    plot_stress_strain_tiled,
    resolve_egi_window,
)
from pyvale.vfm.spatialparambasisfuncs import (
    BasisFunctionKernelBivariate,
    SpatialParameterisationBasisFunction,
)
from pyvale.vfm.spatialparamhomogeneous import SpatialParameterisationHomogeneous


def test_check_stress_against_saved_reports_match_and_difference() -> None:
    saved = np.array([[[[1.0, 2.0]]]])
    computed = saved + 1.0e-10

    check = check_stress_against_saved(computed, saved, rtol=1.0e-8, atol=1.0e-8)

    assert check.saved_stress_available is True
    assert check.matches_saved_stress is True
    assert check.max_abs_difference is not None
    assert np.isclose(check.max_abs_difference, 1.0e-10)


def test_evaluate_snapshot_parameter_maps_rebuilds_additive_bivariate_map() -> None:
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 4))
    homogeneous = SpatialParameterisationHomogeneous(value=250.0)
    basis = SpatialParameterisationBasisFunction(
        x=x,
        y=y,
        kernels=[
            BasisFunctionKernelBivariate(
                x=0.4,
                y=0.6,
                variance_x=0.08,
                variance_y=0.03,
                angle=0.35,
            )
        ],
        heights=[DegreeOfFreedom(40.0, -100.0, 100.0)],
        kernel_type="bivariate",
    )
    snapshot = snapshot_phase(
        {"yield_strength": [homogeneous, basis]}
    )
    experiment_data = SimpleNamespace(
        specimen_geometry=SimpleNamespace(x=x, y=y)
    )

    reconstructed = evaluate_snapshot_parameter_maps(snapshot, experiment_data)

    expected = homogeneous.to_map(np.asarray(x.shape)) + basis.to_map(
        np.asarray(x.shape)
    )
    npt.assert_allclose(reconstructed["yield_strength"], expected)


def test_plot_solve_maps_and_phase_objectives(tmp_path) -> None:
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 2))
    snapshot = snapshot_phase(
        {
            "yield_strength": [SpatialParameterisationHomogeneous(250.0)],
            "hardening_modulus": [SpatialParameterisationHomogeneous(3000.0)],
        }
    )
    result = IdentificationResult(
        parameter_maps={},
        history=IdentificationHistory(
            phases=[
                PhaseResult(
                    phase_index=0,
                    solve_results=[
                        SolveResult(
                            solve_iteration=0,
                            accepted=True,
                            final_objective={"cost": 0.25},
                            final_snapshot=snapshot,
                        ),
                        SolveResult(
                            solve_iteration=1,
                            accepted=False,
                            final_objective={"cost": 0.30},
                            final_snapshot=snapshot,
                        ),
                    ],
                )
            ]
        ),
    )
    roi = SimpleNamespace(
        sample_specimen_mask=lambda grid_x, grid_y: np.ones(
            grid_x.shape,
            dtype=bool,
        )
    )
    experiment_data = SimpleNamespace(
        specimen_geometry=SimpleNamespace(x=x, y=y, region_of_interest=roi)
    )

    solve_maps_path = tmp_path / "solve_maps.png"
    plot_solve_parameter_maps(
        experiment_data,
        result,
        solve_maps_path,
        cmap="viridis",
    )
    plot_phase_objective_histories(result, tmp_path)

    assert solve_maps_path.is_file()
    assert (tmp_path / "objective_history_phase_1.png").is_file()


def test_check_stress_against_saved_handles_missing_reference() -> None:
    check = check_stress_against_saved(np.zeros((1, 3, 2, 2)), None)

    assert check.saved_stress_available is False
    assert check.matches_saved_stress is None


def test_resolve_egi_window_clips_to_odd_available_size() -> None:
    assert resolve_egi_window((10, 12), 29) == (9, 11)
    assert resolve_egi_window((20, 21), (8, 10)) == (7, 9)


def test_compute_parameter_error_diagnostics() -> None:
    identified = {
        "yield_strength": np.array([[110.0, 90.0]]),
        "hardening_modulus": np.array([[12.0, 8.0]]),
    }
    reference = {
        "yield_strength": np.array([[100.0, 100.0]]),
        "hardening_modulus": np.array([[10.0, 10.0]]),
    }

    diagnostics = compute_parameter_error_diagnostics(identified, reference)

    npt.assert_allclose(diagnostics.error_maps["yield_strength"], [[10.0, -10.0]])
    npt.assert_allclose(
        diagnostics.percent_error_maps["hardening_modulus"],
        [[20.0, -20.0]],
    )
    assert diagnostics.summary["yield_strength_max_abs_percent_error"] == 10.0


def test_cache_parameter_error_diagnostics_handles_optional_output(tmp_path) -> None:
    assert cache_parameter_error_diagnostics(tmp_path, None) is None

    diagnostics = ParameterErrorDiagnostics(
        error_maps={"yield_strength": np.array([[10.0]])},
        percent_error_maps={"yield_strength": np.array([[5.0]])},
        summary={},
    )

    paths = cache_parameter_error_diagnostics(tmp_path, diagnostics)

    assert paths is not None
    assert paths[0].exists()
    assert paths[1].exists()


def test_load_constitutive_law_from_result_restores_supported_labels() -> None:
    result = IdentificationResult(
        parameter_maps={},
        metadata=IdentificationMetadata(
            config=ConfigSnapshot(
                constitutive_law=ObjectSnapshot(
                    type_name="IsotropicVonMisesElastoplasticity",
                    module="pyvale.vfm.constlaws",
                    options={
                        "elastic_modulus_label": "E",
                        "poissons_ratio_label": "nu",
                        "hardening_function": {
                            "yield_strength_label": "Y",
                            "hardening_modulus_label": "H",
                        },
                    },
                ),
                hardening_law=ObjectSnapshot(
                    type_name="HardeningLinear",
                    module="pyvale.vfm.hardening",
                    options={
                        "yield_strength_label": "Y",
                        "hardening_modulus_label": "H",
                    },
                ),
            )
        ),
    )

    law = load_constitutive_law_from_result(result)

    assert isinstance(law, IsotropicVonMisesElastoplasticity)
    assert law.elastic_modulus_label == "E"
    assert law.poissons_ratio_label == "nu"
    assert isinstance(law.hardening_function, HardeningLinear)
    assert law.hardening_function.get_required_parameters() == ["Y", "H"]


def test_load_constitutive_law_from_result_restores_compiled_adapter() -> None:
    pytest.importorskip("cython_stress_recon.pyvale_adapter")
    from cython_stress_recon.pyvale_adapter import CompiledLinearHardeningLaw

    result = IdentificationResult(
        parameter_maps={},
        metadata=IdentificationMetadata(
            config=ConfigSnapshot(
                constitutive_law=ObjectSnapshot(
                    type_name="CompiledLinearHardeningLaw",
                    module="cython_stress_recon.pyvale_adapter",
                    options={"error_tolerance": 1.0e-6},
                ),
                hardening_law=ObjectSnapshot(
                    type_name="HardeningLinear",
                    module="pyvale.vfm.hardening",
                    options={},
                ),
            )
        ),
    )

    law = load_constitutive_law_from_result(result)

    assert isinstance(law, CompiledLinearHardeningLaw)
    assert law.error_tolerance == 1.0e-6


def test_parameter_map_summary_uses_ordered_names() -> None:
    summary = parameter_map_summary(
        {
            "hardening_modulus": np.array([[2.0, 4.0]]),
            "yield_strength": np.array([[10.0, 20.0]]),
        }
    )

    assert list(summary)[:3] == [
        "yield_strength_min",
        "yield_strength_mean",
        "yield_strength_max",
    ]


def test_component_history_map_vm_matches_formula() -> None:
    field = np.array(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 3.0], [4.0, 5.0]],
                [[0.5, 0.5], [0.5, 0.5]],
            ]
        ],
        dtype=np.float64,
    )

    vm = component_history_map(field, "vm")
    expected = np.sqrt(
        field[:, 0, :, :] ** 2
        + field[:, 1, :, :] ** 2
        - field[:, 0, :, :] * field[:, 1, :, :]
        + 3.0 * field[:, 2, :, :] ** 2
    )

    npt.assert_allclose(vm, expected)


def test_plot_stress_strain_tiled_writes_output(tmp_path) -> None:
    strain = np.array(
        [
            [
                [[0.10, 0.20], [0.30, 0.40]],
                [[0.11, 0.21], [0.31, 0.41]],
                [[0.01, 0.02], [0.03, 0.04]],
            ],
            [
                [[0.15, 0.25], [0.35, 0.45]],
                [[0.16, 0.26], [0.36, 0.46]],
                [[0.02, 0.03], [0.04, 0.05]],
            ],
        ],
        dtype=np.float64,
    )
    stress = 100.0 * strain
    output_path = tmp_path / "stress_strain_tiled.png"

    plot_stress_strain_tiled(
        strain,
        stress,
        "xx",
        [0, 1],
        [0, 1],
        output_path=output_path,
    )

    assert output_path.exists()
