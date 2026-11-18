"""Shared synthetic ECDF and physical plotting fixtures."""

import numpy as np
import pytest

import pyvale.dataio as io
from pyvale import valid


@pytest.fixture
def ecdf_quantity() -> valid.ValidArray:
    return valid.ValidArray(
        np.array(
            [
                [[0.0, 0.0, 2.0], [2.0, 2.0, 4.0]],
                [[10.0, 10.0, 12.0], [12.0, 12.0, 14.0]],
            ]
        ),
        (
            valid.EValidAxis.EPISTEMIC_TRANS,
            valid.EValidAxis.EPISTEMIC_NUIS,
            valid.EValidAxis.ALEATORY,
        ),
        offsets=valid.EpistemicOffsets(lower=1.0, upper=2.0),
    )


@pytest.fixture
def metric_scene() -> tuple[valid.ValidResult, io.SimData]:
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    observables = {
        "temperature": valid.ValidObservable(
            components=("temperature",),
            geometry=valid.EValidGeometry.POINT,
            coords=coords,
            times=np.array([0.0, 1.0]),
            phys_units="degC",
            coord_units="mm",
            coord_frame="cube",
            time_units="s",
        )
    }
    shape = (4, 1, 2, 2)
    plus = np.broadcast_to(
        np.array([2.0, 0.0, 1.0, np.nan])[:, None, None, None], shape
    ).copy()
    minus = np.broadcast_to(
        np.array([0.0, 3.0, 1.0, np.nan])[:, None, None, None], shape
    ).copy()
    field = valid.ValidMetricField(
        (
            valid.EValidResAxis.POINT,
            valid.EValidResAxis.COMPONENT,
            valid.EValidResAxis.TIME,
            valid.EValidResAxis.SIM_TRANS,
        ),
        {valid.EMetricOutput.D_PLUS: plus, valid.EMetricOutput.D_MINUS: minus},
    )
    result = valid.ValidResult(
        observables,
        {"temperature": {valid.EMetricType.MAVM: field}},
        {valid.EMetricType.MAVM: valid.MetricMAVM()},
    )
    sim_data = io.SimData(
        coords=coords.copy(),
        connect={"connect1": np.array([[1, 2, 3, 4]])},
        time=np.array([0.0, 1.0]),
        num_spat_dims=3,
        node_vars={
            "temperature": np.array(
                [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]]
            )
        },
    )
    return (result, sim_data)
