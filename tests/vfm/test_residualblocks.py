import numpy as np
import pytest

from pyvale.vfm.loadregimes import LoadRegimeThresholds, ResolvedLoadRegimes
from pyvale.vfm.metric import MetricResult
from pyvale.vfm.residualblocks import (
    ResidualBlockSpec,
    prepare_canonical_residual_layout,
)


def _regimes() -> ResolvedLoadRegimes:
    return ResolvedLoadRegimes(
        thresholds=LoadRegimeThresholds(),
        yielded_fraction=(0.0, 0.1, 0.4, 0.8),
        pre_yield=(0,),
        onset=(1,),
        developed=(2,),
        late=(3,),
    )


def test_canonical_blocks_freeze_regimes_masks_and_diagonal_whitening() -> None:
    residual = np.array(
        [
            [100.0, 100.0, 100.0],
            [2.0, np.nan, 6.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
        ]
    )
    result = MetricResult(
        residual=np.zeros_like(residual),
        additional_fields={"normalised_gap": residual},
    )
    layout = prepare_canonical_residual_layout(
        [result],
        _regimes(),
        [
            ResidualBlockSpec(
                name="egi-fine-onset",
                metric_index=0,
                metric_kind="egi",
                load_regime="onset",
                residual_field="normalised_gap",
                physical_support=1.4,
                pixel_support=(7, 7),
                bias=2.0,
                noise_scale=np.array([2.0, 2.0, 4.0]),
                observation_weights=np.array([1.0, 99.0, 3.0]),
                block_weight=2.0,
            )
        ],
    )

    vector = layout.evaluate([result])

    np.testing.assert_allclose(vector.signed, [0.0, 4.0])
    np.testing.assert_allclose(vector.whitened, [0.0, 1.0])
    np.testing.assert_allclose(vector.weighted, [0.0, np.sqrt(1.5)])
    assert vector.block_slice("egi-fine-onset") == slice(0, 2)
    assert layout.blocks[0].coverage_fraction == pytest.approx(2.0 / 3.0)
    assert layout.diagnostics()["blocks"][0]["whitening"] == (
        "diagonal_standard_deviation"
    )


def test_block_influence_is_independent_of_observation_count() -> None:
    results = [
        MetricResult(residual=np.full((2, 1), 2.0)),
        MetricResult(residual=np.full((2, 5), 2.0)),
    ]
    specs = [
        ResidualBlockSpec(
            name=f"block-{index}",
            metric_index=index,
            metric_kind="test",
            load_regime="all",
            noise_scale=2.0,
        )
        for index in range(2)
    ]
    layout = prepare_canonical_residual_layout(results, _regimes(), specs)

    vector = layout.evaluate(results)

    for name in ("block-0", "block-1"):
        block = vector.weighted[vector.block_slice(name)]
        assert np.dot(block, block) == pytest.approx(1.0)


def test_one_dimensional_weights_follow_the_load_step_axis() -> None:
    result = MetricResult(residual=np.ones((4, 2, 3)))
    layout = prepare_canonical_residual_layout(
        [result],
        _regimes(),
        [
            ResidualBlockSpec(
                name="late",
                metric_index=0,
                metric_kind="egi",
                load_regime="late",
                observation_weights=np.array([1.0, 2.0, 3.0, 4.0]),
            )
        ],
    )

    vector = layout.evaluate([result])

    assert vector.weighted.size == 6
    assert np.dot(vector.weighted, vector.weighted) == pytest.approx(1.0)


def test_candidate_must_remain_finite_inside_frozen_mask() -> None:
    reference = MetricResult(residual=np.array([[1.0, np.nan]]))
    layout = prepare_canonical_residual_layout(
        [reference],
        _regimes(),
        [
            ResidualBlockSpec(
                name="guard",
                metric_index=0,
                metric_kind="fre",
                load_regime="all",
            )
        ],
    )
    candidate = MetricResult(residual=np.array([[np.nan, 5.0]]))

    with pytest.raises(ValueError, match="frozen observation mask"):
        layout.evaluate([candidate])
