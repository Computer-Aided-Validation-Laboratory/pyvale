# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""DEVELOPER VERIFICATION MODULE
--------------------------------------------------------------------------------
This module contains developer utility functions used for verification testing
of Directed Acyclic Graph (DAG) sensor error integration in pyvale.
"""

from typing import Callable
import numpy as np

import pyvale.dataio as io
import pyvale.sensorsim as sens
from pyvale.verif import pointsensconst
import pyvale.verif.analyticsimdatafactory as asd


def _calib_assumed(signal: np.ndarray) -> np.ndarray:
    return 24.3 * signal + 0.616


def _calib_truth(signal: np.ndarray) -> np.ndarray:
    return (
        -0.01897
        + 25.41881 * signal
        - 0.42456 * signal**2
        + 0.04365 * signal**3
    )


def _custom_quadratic_op(
    in_state: sens.SignalState,
    error_array: np.ndarray,
    sens_perturbed: sens.SensorData,
) -> sens.SignalState:
    # Custom quadratic transfer function: y = x + 0.001 * x^2
    out_values = in_state.values + 0.001 * (in_state.values**2)
    return sens.SignalState(
        values=out_values,
        sensor_data=sens_perturbed,
    )


def graph_pipeline(
    meas_shape: tuple[int, int, int],
    sens_data: sens.SensorData,
) -> sens.ErrGraph:
    """Topology 1: Multi-Stage Transducer -> Signal Conditioning -> ADC."""
    seed = pointsensconst.GOLD_SEED
    builder = sens.ErrGraphBuilder()
    (
        builder.add_root(
            "thermal_offset",
            sens.ErrSysOffset(offset=1.5),
        )
        .add_child(
            "gain_drift",
            sens.ErrSysGenPercent(
                sens.GenUniform(low=-0.02, high=0.02, seed=seed)
            ),
            parent="thermal_offset",
            op=sens.EErrOp.MULTIPLY,
        )
        .add_child(
            "johnson_noise",
            sens.ErrRandGen(sens.GenNormal(std=0.2, seed=seed)),
            parent="gain_drift",
            op=sens.EErrOp.ADD,
        )
        .add_child(
            "adc_digitize",
            sens.ErrSysDigitisation(bits_per_unit=2**16 / 100),
            parent="johnson_noise",
            op=sens.EErrOp.ADD,
        )
        .add_child(
            "saturation_clamp",
            sens.ErrSysSaturation(meas_min=0.0, meas_max=120.0),
            parent="adc_digitize",
            op=sens.EErrOp.ADD,
        )
    )
    return builder.build(meas_shape, sens_data)


def graph_diamond(
    meas_shape: tuple[int, int, int],
    sens_data: sens.SensorData,
) -> sens.ErrGraph:
    """Topology 2: Diamond Environmental Coupling (Thermal + Vibration)."""
    seed = pointsensconst.GOLD_SEED
    builder = sens.ErrGraphBuilder()
    (
        builder.add_root(
            "root_offset",
            sens.ErrSysOffset(offset=2.0),
        )
        .add_child(
            "branch_thermal",
            sens.ErrSysOffset(offset=1.0),
            parent="root_offset",
        )
        .add_child(
            "branch_vibration",
            sens.ErrRandGen(sens.GenNormal(std=0.5, seed=seed)),
            parent="root_offset",
        )
        .add_node(
            "fanin_saturation",
            sens.ErrSysSaturation(meas_min=0.0, meas_max=100.0),
            inputs=("branch_thermal", "branch_vibration"),
            op=sens.EErrOp.ADD,
        )
    )
    return builder.build(meas_shape, sens_data)


def graph_multichannel_shared(
    meas_shape: tuple[int, int, int],
    sens_data: sens.SensorData,
) -> sens.ErrGraph:
    """Topology 3: Multi-Channel Shared Common-Mode Noise."""
    seed = pointsensconst.GOLD_SEED
    builder = sens.ErrGraphBuilder()
    (
        builder.add_root(
            "ch1_offset",
            sens.ErrSysOffset(offset=-1.0),
        )
        .add_root(
            "ch2_offset",
            sens.ErrSysOffset(offset=2.0),
        )
        .add_node(
            "common_emi",
            sens.ErrRandGen(sens.GenNormal(std=0.8, seed=seed)),
            inputs=("ch1_offset", "ch2_offset"),
            op=sens.EErrOp.ADD,
        )
        .add_child(
            "adc_roundoff",
            sens.ErrSysRoundOff(sens.ERoundMethod.ROUND, 0.05),
            parent="common_emi",
            op=sens.EErrOp.ADD,
        )
    )
    return builder.build(meas_shape, sens_data)


def graph_field_calib(
    meas_shape: tuple[int, int, int],
    sens_data: sens.SensorData,
    field: sens.IField,
) -> sens.ErrGraph:
    """Topology 4: Spatial Field Jitter -> Calibration Inversion -> Noise."""
    seed = pointsensconst.GOLD_SEED

    pos_offset = 0.5 * np.ones_like(sens_data.positions)
    pos_rand = sens.GenNormal(std=0.2, mean=0.0, seed=seed)
    field_err_data = sens.ErrFieldData(
        pos_offset_xyz=pos_offset,
        pos_rand_xyz=(pos_rand, pos_rand, pos_rand),
    )

    builder = sens.ErrGraphBuilder()
    (
        builder.add_root(
            "pos_jitter",
            sens.ErrSysField(field, field_err_data),
        )
        .add_child(
            "calib_curve",
            sens.ErrSysCalibration(
                _calib_assumed,
                _calib_truth,
                cal_range=(0.0, 600.0),
                n_cal_divs=5000,
            ),
            parent="pos_jitter",
            op=sens.EErrOp.ADD,
        )
        .add_child(
            "dependent_noise",
            sens.ErrRandGenPercent(
                sens.GenNormal(std=0.01, seed=seed),
                err_dep=sens.EErrDep.DEPENDENT,
            ),
            parent="calib_curve",
            op=sens.EErrOp.ADD,
        )
    )
    return builder.build(meas_shape, sens_data)


def graph_multitree(
    meas_shape: tuple[int, int, int],
    sens_data: sens.SensorData,
) -> sens.ErrGraph:
    """Topology 5: Multi-Root Multi-Branch Tree Reconvergence."""
    seed = pointsensconst.GOLD_SEED
    builder = sens.ErrGraphBuilder()
    (
        builder.add_root(
            "root_bias1",
            sens.ErrSysOffset(offset=0.5),
        )
        .add_root(
            "root_bias2",
            sens.ErrSysOffset(offset=1.5),
        )
        .add_node(
            "preamp",
            sens.ErrSysGenPercent(
                sens.GenUniform(low=-0.02, high=0.02, seed=seed)
            ),
            inputs=("root_bias1", "root_bias2"),
            op=sens.EErrOp.MULTIPLY,
        )
        .add_child(
            "filter_a",
            sens.ErrRandGen(sens.GenNormal(std=0.3, seed=seed)),
            parent="preamp",
            op=sens.EErrOp.ADD,
        )
        .add_child(
            "filter_b",
            sens.ErrSysOffset(offset=-0.5),
            parent="preamp",
            op=sens.EErrOp.ADD,
        )
        .add_node(
            "daq_sink",
            sens.ErrSysSaturation(meas_min=0.0, meas_max=150.0),
            inputs=("filter_a", "filter_b"),
            op=sens.EErrOp.ADD,
        )
    )
    return builder.build(meas_shape, sens_data)


def graph_custom_op(
    meas_shape: tuple[int, int, int],
    sens_data: sens.SensorData,
) -> sens.ErrGraph:
    """Topology 6: Custom Operator DAG (Quadratic Sensor Transfer Function)."""
    builder = sens.ErrGraphBuilder()
    (
        builder.add_root(
            "baseline_shift",
            sens.ErrSysOffset(offset=2.0),
        )
        .add_child(
            "quadratic_transfer",
            sens.ErrSysOffset(offset=0.0),
            parent="baseline_shift",
            op=sens.EErrOp.CUSTOM,
            custom_op=_custom_quadratic_op,
        )
        .add_child(
            "protection_clamp",
            sens.ErrSysSaturation(meas_min=0.0, meas_max=200.0),
            parent="quadratic_transfer",
            op=sens.EErrOp.ADD,
        )
    )
    return builder.build(meas_shape, sens_data)


def sens_arrays_graph_dict() -> dict[str, sens.SensorsPoint]:
    """Generates a dictionary of SensorsPoint arrays configured with all
    six DAG error graph topologies for gold regression testing.
    """
    sim_data, _ = asd.scalar_linear_2d()
    field = sens.FieldScalar(
        sim_data,
        comp_key="temperature",
        spatial_dims=sens.EDim.TWOD,
    )

    sens_pos = np.array(
        [
            [2.0, 2.0, 0.0],
            [4.0, 3.0, 0.0],
            [6.0, 4.0, 0.0],
            [8.0, 5.0, 0.0],
        ]
    )
    sample_times = np.linspace(0.0, 1.0, 11)
    sens_data = sens.SensorData(
        positions=sens_pos,
        sample_times=sample_times,
    )

    descriptor = sens.DescriptorFactory.temperature()

    type GraphFactory = Callable[
        [tuple[int, int, int], sens.SensorData],
        sens.ErrGraph,
    ]
    graph_factories: dict[str, GraphFactory] = {
        "pipeline": graph_pipeline,
        "diamond": graph_diamond,
        "multichannel": graph_multichannel_shared,
        "multitree": graph_multitree,
        "customop": graph_custom_op,
    }

    sens_dict: dict[str, sens.SensorsPoint] = {}

    for name, factory in graph_factories.items():
        case_key = f"graph_{name}_scal2d"
        sensors = sens.SensorsPoint(sens_data, field, descriptor)
        graph = factory(sensors.get_measurement_shape(), sens_data)
        sensors.set_error_graph(graph)
        sens_dict[case_key] = sensors

    # Spatial field calibration case (needs field parameter)
    case_key = "graph_fieldcalib_scal2d"
    sensors_fc = sens.SensorsPoint(sens_data, field, descriptor)
    graph_fc = graph_field_calib(
        sensors_fc.get_measurement_shape(),
        sens_data,
        field,
    )
    sensors_fc.set_error_graph(graph_fc)
    sens_dict[case_key] = sensors_fc

    return sens_dict
