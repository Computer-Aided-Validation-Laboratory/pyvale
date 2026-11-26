"""Round-trip prepared validation cases, results and metric profiles."""

import json
from pathlib import Path

import numpy as np
import pytest

import pyvale.validation as vl


@pytest.fixture
def archive_case() -> vl.ValidCase:
    observable = vl.ValidObservable(
        components=("temperature",),
        geometry=vl.EValidGeometry.POINT,
        coords=np.array([[1.0, 0.0, 0.0]]),
        connect={"vertex": np.array([[0]], dtype=np.int32)},
        times=np.array([0.0, 2.0]),
        phys_units="degC",
        coord_units="mm",
        coord_frame="plate",
        time_units="s",
    )
    sim = vl.ValidArray(
        np.array([[[1.0, 2.0], [2.0, 3.0]], [[3.0, 4.0], [4.0, 5.0]]]),
        (vl.EValidAxis.TIME, vl.EValidAxis.EPISTEMIC_NUIS, vl.EValidAxis.ALEATORY),
    )
    exp = vl.ValidArray(
        np.array([[2.0, 3.0], [np.nan, np.nan]]),
        (vl.EValidAxis.TIME, vl.EValidAxis.ALEATORY),
        offsets=vl.EpistemicOffsets(
            np.array([0.5, 1.0]), np.array([1.5, 2.0]), (vl.EValidAxis.TIME,)
        ),
    )
    nuisance_axis = (vl.EValidAxis.EPISTEMIC_NUIS,)
    return vl.ValidCase(
        {"temperature": observable},
        {"temperature": sim},
        {"temperature": exp},
        sim_inputs=vl.ValidSampleInputs(
            input_params={"gain": vl.ValidArray(np.array([1, 2]), nuisance_axis)},
            run_ids=vl.ValidArray(np.array(["run_a", "run_b"]), nuisance_axis),
        ),
    )


def test_valid_case_round_trip(archive_case: vl.ValidCase, tmp_path: Path) -> None:
    path = tmp_path / "case.npz"
    vl.save_valid_case(path, archive_case)
    restored = vl.load_valid_case(path)

    tag = "temperature"
    for original, loaded in (
        (archive_case.sim[tag], restored.sim[tag]),
        (archive_case.exp[tag], restored.exp[tag]),
    ):
        assert loaded.axes == original.axes
        np.testing.assert_array_equal(loaded.data, original.data)
    exp_offsets = restored.exp[tag].offsets
    assert exp_offsets is not None
    assert exp_offsets.axes == (vl.EValidAxis.TIME,)
    np.testing.assert_array_equal(exp_offsets.lower, [0.5, 1.0])
    np.testing.assert_array_equal(exp_offsets.upper, [1.5, 2.0])
    assert restored.sim[tag].offsets is None
    assert restored.exp_inputs is None
    assert restored.sim_inputs is not None
    assert restored.sim_inputs.input_params is not None
    assert restored.sim_inputs.run_ids is not None
    np.testing.assert_array_equal(restored.sim_inputs.input_params["gain"].data, [1, 2])
    np.testing.assert_array_equal(restored.sim_inputs.run_ids.data, ["run_a", "run_b"])

    observable = restored.observables[tag]
    assert observable.geometry is vl.EValidGeometry.POINT
    assert observable.components == ("temperature",)
    assert observable.coord_frame == "plate"
    assert (observable.phys_units, observable.coord_units, observable.time_units) == (
        "degC",
        "mm",
        "s",
    )
    assert observable.connect is not None
    assert observable.connect["vertex"].dtype == np.int32
    np.testing.assert_array_equal(observable.coords, [[1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(observable.times, [0.0, 2.0])


@pytest.mark.parametrize("legacy_offset_key", [False, True])
def test_result_and_extrema_round_trip(
    archive_case: vl.ValidCase, tmp_path: Path, legacy_offset_key: bool
) -> None:
    metrics = [
        vl.MetricMAVM(alpha=0.2, duplicates=vl.EMAVMDupes.REJECT, tol=1e-9),
        vl.MetricRelErr(normalisation=vl.ERelErrNorm.CONSTANT, constant=10.0),
    ]
    original = vl.ValidEngine(metrics).run(archive_case)
    result_path = tmp_path / "result.npz"
    vl.save_valid_result(result_path, original)
    with np.load(result_path, allow_pickle=False) as archive:
        manifest = json.loads(archive["manifest"].item())
        arrays = {key: archive[key] for key in archive.files if key != "manifest"}
    for fields in manifest["metadata"]["metrics"].values():
        for stored in fields.values():
            assert "epis_offsets" in stored and "scenarios" not in stored
            if legacy_offset_key:
                stored["scenarios"] = stored.pop("epis_offsets")
    if legacy_offset_key:
        np.savez(result_path, manifest=np.array(json.dumps(manifest)), **arrays)
    restored = vl.load_valid_result(result_path)

    tag = "temperature"
    for kind, original_field in original.metrics[tag].items():
        loaded = restored.metrics[tag][kind]
        assert loaded.axes == original_field.axes
        assert loaded.epis_offsets == original_field.epis_offsets
        assert loaded.metric_vals.keys() == original_field.metric_vals.keys()
        for output in loaded.metric_vals:
            np.testing.assert_array_equal(
                loaded.metric_vals[output], original_field.metric_vals[output]
            )
    mavm = restored.metric_configs[vl.EMetricType.MAVM]
    assert isinstance(mavm, vl.MetricMAVM)
    assert (mavm.alpha, mavm.duplicates, mavm.tol) == (
        0.2,
        vl.EMAVMDupes.REJECT,
        1e-9,
    )
    relative_error = restored.metric_configs[vl.EMetricType.REL_ERR]
    assert isinstance(relative_error, vl.MetricRelErr)
    assert (relative_error.normalisation, relative_error.constant) == (
        vl.ERelErrNorm.CONSTANT,
        10.0,
    )
    np.testing.assert_array_equal(restored.observables[tag].times, [0.0, 2.0])

    profile = vl.profile_epis_nuisance_extrema(
        restored,
        tag,
        vl.EMetricType.MAVM,
        selectors={vl.EValidResAxis.EXP_OFFSET: vl.EEpisOffset.NOMINAL},
    )
    extrema_path = tmp_path / "extrema.npz"
    vl.save_metric_extrema(extrema_path, profile)
    extrema = vl.load_metric_extrema(extrema_path)
    assert extrema.axes == profile.axes
    assert extrema.reduced_axes == profile.reduced_axes
    assert extrema.selectors == profile.selectors
    for output in profile.minimum:
        np.testing.assert_array_equal(extrema.minimum[output], profile.minimum[output])
        np.testing.assert_array_equal(extrema.maximum[output], profile.maximum[output])
        np.testing.assert_array_equal(
            extrema.minimum_indices[output], profile.minimum_indices[output]
        )
        np.testing.assert_array_equal(
            extrema.maximum_indices[output], profile.maximum_indices[output]
        )


def test_load_rejects_wrong_archive_kind_and_version(
    archive_case: vl.ValidCase, tmp_path: Path
) -> None:
    path = tmp_path / "archive.npz"
    vl.save_valid_case(path, archive_case)
    with pytest.raises(ValueError, match="Expected a result"):
        vl.load_valid_result(path)

    np.savez(path, manifest=np.array('{"version": 999, "kind": "case"}'))
    with pytest.raises(ValueError, match="schema version"):
        vl.load_valid_case(path)


def test_save_rejects_object_arrays(archive_case: vl.ValidCase, tmp_path: Path) -> None:
    assert archive_case.sim_inputs is not None
    archive_case.sim_inputs.run_ids = vl.ValidArray(
        np.array(["run_a", "run_b"], dtype=object), (vl.EValidAxis.EPISTEMIC_NUIS,)
    )
    path = tmp_path / "case.npz"
    with pytest.raises(TypeError, match="object arrays"):
        vl.save_valid_case(path, archive_case)
    assert not path.exists()
