# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Tests for discrete site and separation constraints."""

import numpy as np
import pytest

import pyvale.sensopt as opt
import pyvale.sensorsim as sens


def test_pair_distances_replace_global_spacing() -> None:
    constraint = opt.ConstrainSensSep(
        min_distance=10.0,
        pair_distances={
            ("temperature", "temperature"): 5.0,
            ("temperature", "strain"): 3.0,
        },
    )
    temperature = sens.SensorData(
        positions=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    )
    strain = sens.SensorData(positions=np.array([[8.0, 0.0, 0.0]]))
    residuals = constraint.eval({"temperature": temperature, "strain": strain})
    assert len(residuals) == 3
    assert np.all(residuals <= 0)

    reference = sens.SensorData(positions=np.array([[12.0, 0.0, 0.0]]))
    assert (
        np.max(
            constraint.eval(
                {"temperature": temperature, "strain": strain, "reference": reference}
            )
        )
        == 6.0
    )


def test_conflicting_pair_distances_are_rejected() -> None:
    constraint = opt.ConstrainSensSep(
        pair_distances={("first", "second"): 1.0, ("second", "first"): 2.0}
    )
    with pytest.raises(ValueError, match="Conflicting separation distances"):
        constraint.eval({})


def test_site_selectors_combine_and_override_default() -> None:
    grid_xyz = np.stack(
        np.meshgrid(np.arange(4.0), np.arange(3.0), np.array([0.0]), indexing="ij"),
        axis=-1,
    )
    outer = opt.SensSiteBounds(x=(1.0, 3.0))
    constraint = opt.ConstrainSensSites(
        default=outer,
        by_array={
            "temperature": opt.SensSiteAll(
                selectors=(
                    outer,
                    opt.SensSiteNot(selector=opt.SensSiteBounds(y=(1.0, 1.0))),
                )
            ),
            "strain": opt.SensSiteAny(
                selectors=(
                    opt.SensSiteGridIdxs(x=(0,), y=range(1, 3)),
                    np.array([[3.0, 0.0, 0.0]]),
                )
            ),
            "disabled": np.empty((0, 3)),
        },
        grid_xyz=grid_xyz,
    )
    # Filtered and reordered rows still refer to the original grid indices.
    positions = grid_xyz.reshape(-1, 3)[[6, 1, 9, 7]]
    sensor_data = sens.SensorData(positions=positions)
    np.testing.assert_array_equal(
        constraint.eval({"reference": sensor_data}), [0, 1, 0, 0]
    )
    np.testing.assert_array_equal(
        constraint.eval({"temperature": sensor_data}), [0, 1, 0, 1]
    )
    np.testing.assert_array_equal(
        constraint.eval({"strain": sensor_data}), [1, 0, 0, 1]
    )
    np.testing.assert_array_equal(
        constraint.eval({"disabled": sensor_data}), [1, 1, 1, 1]
    )


def test_grid_indices_require_the_original_grid() -> None:
    selector = opt.SensSiteNot(selector=opt.SensSiteGridIdxs(x=(1,)))
    with pytest.raises(ValueError, match="grid_xyz is required"):
        opt.ConstrainSensSites(default=selector)
    with pytest.raises(ValueError, match="within the grid"):
        opt.ConstrainSensSites(default=selector, grid_xyz=np.zeros((1, 1, 1, 3)))
