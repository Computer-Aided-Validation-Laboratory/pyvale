# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Tests for Directed Acyclic Graph (DAG) sensor error integration."""

import numpy as np
import pytest

import pyvale.sensorsim as sens


def test_err_graph_linear_equivalence() -> None:
    """A linear ErrGraph produces identical results to ErrIntegrator."""
    meas_shape = (4, 1, 10)
    seed = 12345
    truth = np.ones(meas_shape) * 10.0

    sens_data = sens.SensorData(
        positions=np.zeros((4, 3)),
        sample_times=np.linspace(0.0, 1.0, 10),
    )

    errs_chain = [
        sens.ErrSysOffset(offset=2.0),
        sens.ErrRandGen(sens.GenNormal(std=0.5, seed=seed)),
        sens.ErrSysSaturation(meas_min=0.0, meas_max=11.0),
    ]

    # Evaluate classic ErrIntegrator
    integrator = sens.ErrIntegrator(
        err_chain=errs_chain,
        sensor_data_initial=sens_data,
        meas_shape=meas_shape,
    )
    integrator.reseed_error_chain(seed)
    chain_total = integrator.calc_errors_from_chain(truth)
    chain_sys = integrator.get_errs_systematic()
    chain_rand = integrator.get_errs_random()

    # Evaluate ErrGraph created via helper
    graph = sens.err_chain_to_graph(
        err_chain=errs_chain,
        meas_shape=meas_shape,
        sensor_data_initial=sens_data,
    )
    graph.reseed(seed)
    graph_total = graph.calc_errors_from_graph(truth)
    graph_sys = graph.get_errs_systematic()
    graph_rand = graph.get_errs_random()

    assert np.allclose(chain_total, graph_total)
    assert np.allclose(chain_sys, graph_sys)
    assert np.allclose(chain_rand, graph_rand)


def test_err_graph_cycle_detection() -> None:
    """ErrGraph raises ValueError when given cyclic node dependencies."""
    meas_shape = (2, 1, 5)
    sens_data = sens.SensorData(positions=np.zeros((2, 3)))

    nodes = [
        sens.ErrNode("node_a", sens.ErrSysOffset(1.0), inputs=("node_c",)),
        sens.ErrNode("node_b", sens.ErrSysOffset(2.0), inputs=("node_a",)),
        sens.ErrNode("node_c", sens.ErrSysOffset(3.0), inputs=("node_b",)),
    ]

    with pytest.raises(ValueError, match="cycle"):
        sens.ErrGraph(nodes, meas_shape, sens_data)


def test_err_graph_missing_input_raises() -> None:
    """ErrGraph raises KeyError when an input node name does not exist."""
    meas_shape = (2, 1, 5)
    sens_data = sens.SensorData(positions=np.zeros((2, 3)))

    nodes = [
        sens.ErrNode(
            "node_a", sens.ErrSysOffset(1.0), inputs=("non_existent",)
        ),
    ]

    with pytest.raises(KeyError, match="non_existent"):
        sens.ErrGraph(nodes, meas_shape, sens_data)


def test_err_graph_duplicate_names_raises() -> None:
    """ErrGraph raises ValueError when node names are duplicated."""
    meas_shape = (2, 1, 5)
    sens_data = sens.SensorData(positions=np.zeros((2, 3)))

    nodes = [
        sens.ErrNode("duplicate", sens.ErrSysOffset(1.0)),
        sens.ErrNode("duplicate", sens.ErrSysOffset(2.0)),
    ]

    with pytest.raises(ValueError, match="unique"):
        sens.ErrGraph(nodes, meas_shape, sens_data)


def test_err_graph_diamond_dag() -> None:
    """ErrGraph evaluates diamond DAG dependencies with multiple branches."""
    meas_shape = (2, 1, 5)
    truth = np.full(meas_shape, 5.0)
    sens_data = sens.SensorData(positions=np.zeros((2, 3)))

    # Diamond graph structure:
    #         root_offset (+2.0)
    #           /          \
    # branch_a (+1.0)    branch_b (+3.0)
    #           \          /
    #         sink_sat (max 10.0)
    builder = sens.ErrGraphBuilder()
    (
        builder.add_root("root_offset", sens.ErrSysOffset(offset=2.0))
        .add_child(
            "branch_a", sens.ErrSysOffset(offset=1.0), parent="root_offset"
        )
        .add_child(
            "branch_b", sens.ErrSysOffset(offset=3.0), parent="root_offset"
        )
        .add_node(
            "sink_sat",
            sens.ErrSysSaturation(meas_min=0.0, meas_max=10.0),
            inputs=("branch_a", "branch_b"),
        )
    )

    opts = sens.ErrGraphOpts(store_node_outputs=True)
    graph = builder.build(meas_shape, sens_data, opts=opts)
    total_err = graph.calc_errors_from_graph(truth)

    # truth = 5.0
    # root_offset: out = 5 + 2 = 7.0
    # branch_a: out = 7 + 1 = 8.0 (dev = +3 from truth)
    # branch_b: out = 7 + 3 = 10.0 (dev = +5 from truth)
    # sink input values = 5.0 + 3.0 + 5.0 = 13.0
    # sink_sat: capped to 10.0 -> total err = 10.0 - 5.0 = 5.0
    assert np.allclose(truth + total_err, 10.0)

    node_outputs = graph.get_node_outputs()
    assert node_outputs is not None
    assert "branch_a" in node_outputs
    assert np.allclose(node_outputs["branch_a"].values, 8.0)
    assert np.allclose(node_outputs["branch_b"].values, 10.0)


def test_err_graph_multiplication_and_replace() -> None:
    """ErrGraph supports EErrOp.MULTIPLY and EErrOp.REPLACE."""
    meas_shape = (2, 1, 4)
    truth = np.full(meas_shape, 10.0)
    sens_data = sens.SensorData(positions=np.zeros((2, 3)))

    nodes = [
        sens.ErrNode(
            "gain_error",
            sens.ErrSysOffset(offset=0.1),  # 10% gain factor
            inputs=(),
            op=sens.EErrOp.MULTIPLY,
        ),
        sens.ErrNode(
            "clipper",
            sens.ErrSysSaturation(meas_min=0.0, meas_max=10.5),
            inputs=("gain_error",),
            op=sens.EErrOp.ADD,
        ),
    ]

    graph = sens.ErrGraph(nodes, meas_shape, sens_data)
    total_err = graph.calc_errors_from_graph(truth)

    # gain_error (MULTIPLY): 10.0 * (1 + 0.1) = 11.0
    # clipper (ADD): sat(11.0, max=10.5) - 11.0 = -0.5 -> 11.0 - 0.5 = 10.5
    assert np.allclose(truth + total_err, 10.5)


def test_sensors_point_with_err_graph() -> None:
    """SensorsPoint integrates cleanly with an ErrGraph."""
    import pyvale.verif.analyticsimdatafactory as asd
    from pyvale.sensorsim.fieldscalar import FieldScalar

    sim_data, _ = asd.scalar_linear_2d()
    field = FieldScalar(
        sim_data,
        comp_key="temperature",
        spatial_dims=sens.EDim.TWOD,
    )

    sens_pos = np.array([[2.0, 2.0, 0.0]])
    sens_data = sens.SensorData(
        positions=sens_pos,
        sample_times=np.array([0.0]),
    )
    sensors = sens.SensorsPoint(sens_data, field)

    builder = sens.ErrGraphBuilder()
    builder.add_root("offset", sens.ErrSysOffset(offset=5.0))
    graph = builder.build(sensors.get_measurement_shape(), sens_data)

    sensors.set_error_graph(graph)
    truth = sensors.get_truth()
    meas = sensors.sim_measurements()

    assert np.allclose(meas, truth + 5.0)
    assert np.allclose(sensors.get_errors_systematic(), 5.0)
    assert np.allclose(sensors.get_errors_random(), 0.0)
