from pathlib import Path

import numpy as np
import numpy.testing as npt
import yaml

from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.identificationresult import (
    IdentificationHistory,
    IdentificationResult,
    ParameterisationSnapshot,
    PhaseResult,
    load_identification_result,
    snapshot_phase,
    snapshot_refinement_action,
    snapshot_refinement_policy,
    summarise_parameterisation,
)
from pyvale.vfm.refinement import SliceMergeSplitAction, SliceMergeSplitRefinement
from pyvale.vfm.spatialparambasisfuncs import (
    BasisFunctionKernelBivariate,
    SpatialParameterisationBasisFunction,
)
from pyvale.vfm.spatialparamslicewise import (
    SliceConfig,
    SupportSlice,
    SliceWiseSpatialParameterisation,
)


def test_identification_result_bundle_round_trips_maps_stress_and_history(
    tmp_path: Path,
) -> None:
    parameter_maps = {
        "yield_strength": np.array([[250.0, 275.0], [300.0, 325.0]]),
        "hardening_modulus": np.full((2, 2), 7_000.0),
    }
    final_stress = np.arange(24, dtype=np.float64).reshape(2, 3, 2, 2)
    parameterisation = SliceWiseSpatialParameterisation(
        slice_config=SliceConfig(
            axis="x",
            boundaries=np.array([0.0, 1.0, 2.0]),
        ),
        values=[
            DegreeOfFreedom(250.0, 100.0, 500.0),
            DegreeOfFreedom(325.0, 100.0, 500.0),
        ],
    )
    result = IdentificationResult(
        parameter_maps=parameter_maps,
        history=IdentificationHistory(
            phases=[
                PhaseResult(
                    phase_index=0,
                    final_snapshot=snapshot_phase(
                        {"yield_strength": [parameterisation]}
                    ),
                )
            ]
        ),
        final_stress=final_stress,
    )

    result_file = result.save_to_yaml(tmp_path)
    loaded = load_identification_result(result_file)

    assert (tmp_path / "identification_result.yaml").is_file()
    assert (tmp_path / "final_parameter_maps.npz").is_file()
    assert (tmp_path / "final_identified_stress.npz").is_file()
    npt.assert_allclose(
        loaded.parameter_maps["yield_strength"],
        parameter_maps["yield_strength"],
    )
    npt.assert_allclose(loaded.final_stress, final_stress)

    loaded_snapshot = (
        loaded.history.phases[0]
        .spatial_parameterisations["yield_strength"][0]
    )
    assert loaded_snapshot.parameterisation is None
    assert loaded_snapshot.parameterisation_type == "SliceWiseSpatialParameterisation"
    assert loaded_snapshot.summary["kind"] == "slice_wise"
    assert loaded_snapshot.summary["axis"] == "x"
    assert loaded_snapshot.summary["boundaries"] == [0.0, 1.0, 2.0]


def test_basis_parameterisation_summary_stores_literal_kernel_geometry() -> None:
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    parameterisation = SpatialParameterisationBasisFunction(x, y)
    parameterisation.kernels.append(
        BasisFunctionKernelBivariate(
            x=0.25,
            y=0.75,
            variance_x=0.04,
            variance_y=0.09,
            angle=0.5,
        )
    )
    parameterisation.heights.append(DegreeOfFreedom(120.0, -200.0, 200.0))

    summary = summarise_parameterisation(parameterisation)

    assert summary["kind"] == "basis_functions"
    assert summary["num_kernels"] == 1
    kernel = summary["kernels"][0]
    assert kernel["centre"] == [0.25, 0.75]
    assert kernel["variance"] == [0.04, 0.09]
    assert kernel["width"] == [0.2, 0.3]
    assert kernel["angle"] == 0.5
    assert kernel["height"] == 120.0


def test_refinement_snapshots_avoid_runtime_support_graph() -> None:
    support = SupportSlice(slice_config=SliceConfig(axis="x", num_slices=4))
    policy = SliceMergeSplitRefinement(
        target=support,
        max_refinements=2,
        merge_parameter_tolerance=0.05,
        split_error_threshold=0.2,
    )
    action = SliceMergeSplitAction(
        support=support,
        refined_boundaries=np.array([0.0, 1.0, 2.0]),
        policy=policy,
    )

    policy_snapshot = snapshot_refinement_policy(policy)
    action_snapshot = snapshot_refinement_action(action)

    assert policy_snapshot.type_name == "SliceMergeSplitRefinement"
    assert policy_snapshot.options == {
        "max_refinements": 2,
        "merge_parameter_tolerance": 0.05,
        "split_error_threshold": 0.2,
    }
    assert "target" not in policy_snapshot.options
    assert action_snapshot.type_name == "SliceMergeSplitAction"
    assert action_snapshot.options == {}


def test_load_from_yaml_accepts_previous_parameter_map_file_layout(
    tmp_path: Path,
) -> None:
    parameter_map = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.save(tmp_path / "parameter_map_yield_strength.npy", parameter_map)
    (tmp_path / "identification_result.yaml").write_text(
        yaml.safe_dump(
            {
                "parameter_maps": {
                    "yield_strength": "parameter_map_yield_strength.npy",
                },
                "history": {
                    "phases": [
                        {
                            "spatial_parameterisations": {
                                "yield_strength": [
                                    {
                                        "parameterisation": "LegacyType",
                                        "dof_values": [1.0],
                                    }
                                ]
                            }
                        }
                    ]
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = load_identification_result(tmp_path)

    npt.assert_allclose(result.parameter_maps["yield_strength"], parameter_map)
    snapshot: ParameterisationSnapshot = (
        result.history.phases[0]
        .spatial_parameterisations["yield_strength"][0]
    )
    assert snapshot.parameterisation_type == "LegacyType"
    assert snapshot.dof_values == [1.0]
