"""Tests for generic workflow composition and execution."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import pyvale.workflow as workflow


def verify_initial(case, context, input_data) -> None:
    """Verify the initial empty pipeline payload."""
    assert input_data is None


def make_value(case, context, input_data) -> dict[str, float]:
    """Make a deterministic metric from one case."""
    return {"value": float(case.values["number"])}


def verify_vsg(case) -> None:
    """Reject an invalid generic study constraint."""
    if case.values["number"] < 0:
        raise ValueError("number must be non-negative")


def test_factorial_supports_numeric_and_categorical_values() -> None:
    """Factorial generation preserves numeric and categorical values."""
    design = workflow.FullFactorial((
        workflow.ParameterValues("number", workflow.EParameterKind.NUMERIC, (1, 2)),
        workflow.ParameterValues(
            "shape", workflow.EParameterKind.CATEGORICAL, ("AFFINE",),
        ),
    ))

    cases = tuple(design.generate(42))

    assert design.count() == 2
    assert cases[1].values == {"number": 2, "shape": "AFFINE"}
    assert cases[0].seed != cases[1].seed


def test_random_sampling_is_reproducible() -> None:
    """Random case designs use reproducible NumPy generators."""
    design = workflow.RandomSampling(
        {"number": lambda rng: int(rng.integers(1, 10))},
        count=3,
    )

    first = tuple(design.generate(13))
    second = tuple(design.generate(13))

    assert [case.values for case in first] == [case.values for case in second]


def test_pipeline_runner_persists_and_gathers_results(tmp_path: Path) -> None:
    """Generic function steps run, persist summaries, and gather columns."""
    design = workflow.ExplicitCases(({"number": 1}, {"number": 2}))
    cases = list(design.generate(3))
    pipeline = workflow.PipelineWorkflow((
        workflow.FunctionStep(verify_initial, make_value),
    ), verify_vsg)
    results = workflow.WorkflowRunner(
        workflow.WorkflowConfig(tmp_path, workers=1),
    ).run(pipeline, cases)

    dataset = workflow.WorkflowGatherer.gather(tmp_path)

    assert [result.metrics["value"] for result in results] == [1.0, 2.0]
    assert dataset.parameters["number"].tolist() == [1, 2]
    assert dataset.metrics["value"].tolist() == [1.0, 2.0]
    assert (tmp_path / "summary.npz").is_file()


def test_repeat_aggregation_and_signal_extraction() -> None:
    """Gathering aggregates repeats and ignores non-finite selected values."""
    cases = tuple(workflow.ExplicitCases((
        {"level": 1, "repeat": 0},
        {"level": 1, "repeat": 1},
    )).generate(4))
    results = tuple(
        workflow.CaseResult(
            case, {"noise_floor": float(index + 1)}, (),
            workflow.ECaseStatus.COMPLETED, 0.0,
        )
        for index, case in enumerate(cases)
    )
    dataset = workflow.WorkflowGatherer.from_results(results)
    aggregated = workflow.WorkflowGatherer.aggregate_repeats(dataset)
    values = np.array(((1.0, np.nan), (3.0, 5.0)))
    pixels_x, pixels_y = np.meshgrid(np.arange(2), np.arange(2))
    extraction = workflow.SignalExtraction(
        workflow.EStrainComponent.EPS_XX,
        workflow.FullFieldSelector(),
    )

    assert aggregated.parameters["level"].tolist() == [1]
    assert aggregated.metrics["noise_floor_mean"].tolist() == [1.5]
    assert extraction.reduce(values, pixels_x, pixels_y) == 3.0


def test_pipeline_runner_uses_multiprocessing(tmp_path: Path) -> None:
    """Spawned workers preserve ordered deterministic workflow results."""
    cases = list(
        workflow.ExplicitCases(({"number": 1}, {"number": 2})).generate(5),
    )
    pipeline = workflow.PipelineWorkflow((
        workflow.FunctionStep(verify_initial, make_value),
    ), verify_vsg)

    results = workflow.WorkflowRunner(
        workflow.WorkflowConfig(tmp_path, workers=2),
    ).run(pipeline, cases)

    assert [result.metrics["value"] for result in results] == [1.0, 2.0]


def test_noise_preserves_dtype_and_is_reproducible() -> None:
    """Grey-level noise uses the supplied deterministic random generator."""
    image = np.full((16, 16), 128, dtype=np.uint8)
    first = workflow.add_grey_level_noise(image, np.random.default_rng(7))
    second = workflow.add_grey_level_noise(image, np.random.default_rng(7))

    assert first.dtype == image.dtype
    assert np.array_equal(first, second)
    assert np.std(first.astype(float) - image.astype(float)) > 0.5


def test_runner_removes_large_artifacts_when_requested(tmp_path: Path) -> None:
    """Artifact cleanup preserves compact case manifests."""
    def write_artifact(case, context, input_data):
        path = context.case_dir / "large.tiff"
        path.write_bytes(b"image")
        context.add_artifact(path)
        return {"value": 1.0}

    pipeline = workflow.PipelineWorkflow((
        workflow.FunctionStep(verify_initial, write_artifact),
    ))
    case = next(workflow.ExplicitCases(({"number": 1},)).generate(1))
    workflow.WorkflowRunner(
        workflow.WorkflowConfig(tmp_path, retain_artifacts=False),
    ).run(pipeline, [case])

    case_dir = tmp_path / "cases" / "000000"
    assert (case_dir / "parameters.json").is_file()
    assert (case_dir / "summary.json").is_file()
    assert not (case_dir / "large.tiff").exists()


def test_memory_storage_removes_case_files(tmp_path: Path) -> None:
    """Memory mode removes temporary case data after collecting results."""
    pipeline = workflow.PipelineWorkflow((
        workflow.FunctionStep(verify_initial, make_value),
    ), verify_vsg)
    case = next(workflow.ExplicitCases(({"number": 1},)).generate(1))

    results = workflow.WorkflowRunner(
        workflow.WorkflowConfig(
            tmp_path,
            storage=workflow.EWorkflowStorage.MEMORY,
        ),
    ).run(pipeline, [case])

    assert results[0].metrics == {"value": 1.0}
    assert not (tmp_path / "cases" / "000000").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_pixel_selectors_return_expected_values() -> None:
    """Point, line, area, and whole-field selectors use pixel coordinates."""
    pixels_x, pixels_y = np.meshgrid(np.arange(3), np.arange(3))
    values = pixels_x + 10 * pixels_y

    point = workflow.PointSelector(np.array((2.0, 1.0)))
    area = workflow.AreaSelector(np.array((1.0, 1.0)), np.array((2.0, 2.0)))

    assert point.select(values, pixels_x, pixels_y).tolist() == [12]
    assert sorted(area.select(values, pixels_x, pixels_y).tolist()) == [
        11, 12, 21, 22,
    ]
    whole_field = workflow.FullFieldSelector().select(values, pixels_x, pixels_y)
    assert whole_field.size == 9
    mask = np.eye(3, dtype=bool)
    selected = workflow.MaskSelector(mask).select(values, pixels_x, pixels_y)
    assert selected.tolist() == [0, 11, 22]
