"""Shared synthetic ECDF and physical plotting fixtures."""

import numpy as np
import pytest

import pyvale.dataio as io
import pyvale.validation as vl


@pytest.fixture
def ecdf_quantity() -> vl.ValidArray:
    return vl.ValidArray(
        np.array(
            [
                [[0.0, 0.0, 2.0], [2.0, 2.0, 4.0]],
                [[10.0, 10.0, 12.0], [12.0, 12.0, 14.0]],
            ]
        ),
        (
            vl.EValidAxis.EPISTEMIC_TRANS,
            vl.EValidAxis.EPISTEMIC_NUIS,
            vl.EValidAxis.ALEATORY,
        ),
        offsets=vl.EpistemicOffsets(lower=1.0, upper=2.0),
    )


@pytest.fixture
def metric_scene() -> tuple[vl.ValidResult, io.SimData]:
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    observables = {
        "temperature": vl.ValidObservable(
            components=("temperature",),
            geometry=vl.EValidGeometry.POINT,
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
    field = vl.MetricField(
        (
            vl.EValidResAxis.POINT,
            vl.EValidResAxis.COMPONENT,
            vl.EValidResAxis.TIME,
            vl.EValidResAxis.SIM_TRANS,
        ),
        {vl.EMetricOut.D_PLUS: plus, vl.EMetricOut.D_MINUS: minus},
    )
    result = vl.ValidResult(
        observables,
        {"temperature": {vl.EMetricType.MAVM: field}},
        {vl.EMetricType.MAVM: vl.MetricMAVM()},
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
