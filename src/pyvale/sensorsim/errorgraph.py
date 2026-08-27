# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Directed Acyclic Graph (DAG) error integration engine for virtual sensors.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import enum
from graphlib import CycleError, TopologicalSorter
from typing import Callable, Sequence

import numpy as np

from pyvale.sensorsim.errorsimulator import EErrDep, EErrType, IErrSimulator
from pyvale.sensorsim.sensordata import SensorData


class EErrOp(enum.Enum):
    """Operation used to combine node outputs with incoming dependencies."""

    ADD = enum.auto()
    """Output values = input_values + error_array."""

    MULTIPLY = enum.auto()
    """Output values = input_values * (1.0 + error_array)."""

    REPLACE = enum.auto()
    """Output values = error_array (e.g. for non-linear quantization)."""

    CUSTOM = enum.auto()
    """User-defined custom combination callable."""


@dataclass(slots=True)
class SignalState:
    """Carries measurement values and sensor geometric parameters through
    the graph.
    """

    values: np.ndarray
    """Array of signal or measurement values with shape=(num_sensors,
    num_components, num_times).
    """

    sensor_data: SensorData
    """SensorData object containing sensor positions, angles, and sample times.
    """


@dataclass(slots=True)
class ErrNode:
    """A single computation node in the sensor error DAG."""

    name: str
    """Unique identifier for this node."""

    simulator: IErrSimulator
    """Error simulator implementing the IErrSimulator interface."""

    inputs: tuple[str, ...] = ()
    """Names of parent nodes that feed into this node. An empty tuple means
    the node is a root that receives the nominal ground-truth state.
    """

    op: EErrOp = EErrOp.ADD
    """Combination operation to apply to simulator output."""

    custom_op: (
        Callable[[SignalState, np.ndarray, SensorData], SignalState] | None
    ) = None
    """Custom operator callable when op == EErrOp.CUSTOM."""


@dataclass(slots=True)
class ErrGraphOpts:
    """Options controlling error graph execution and caching."""

    store_node_outputs: bool = False
    """Stores all intermediate SignalState objects for all nodes if True."""

    force_dependence: EErrDep | None = None
    """Forces all error simulators in the graph to use the specified dependence.
    """


class ErrGraph:
    """Directed Acyclic Graph (DAG) sensor error integration engine.

    Evaluates error simulators in topological dependency order, supporting
    branching, non-additive combination operations, and per-node error
    attribution.
    """

    __slots__ = (
        "_nodes",
        "_execution_order",
        "_meas_shape",
        "_sens_data_initial",
        "_sens_data_accumulated",
        "_opts",
        "_errs_systematic",
        "_errs_random",
        "_errs_total",
        "_node_outputs",
        "_node_errors",
    )

    def __init__(
        self,
        nodes: Sequence[ErrNode],
        meas_shape: tuple[int, int, int],
        sensor_data_initial: SensorData,
        opts: ErrGraphOpts | None = None,
    ) -> None:
        """
        Parameters
        ----------
        nodes : Sequence[ErrNode]
            Collection of graph nodes defining error calculators and edges.
        meas_shape : tuple[int, int, int]
            Shape of measurement array (num_sensors, num_components, num_times).
        sensor_data_initial : SensorData
            Initial nominal sensor parameters.
        opts : ErrGraphOpts | None, optional
            Options controlling graph execution, by default None.
        """
        self._nodes: dict[str, ErrNode] = {n.name: n for n in nodes}
        if len(self._nodes) != len(nodes):
            raise ValueError(
                "All ErrNode instances in ErrGraph must have unique names."
            )

        self._meas_shape = meas_shape
        self._sens_data_initial = copy.deepcopy(sensor_data_initial)
        self._sens_data_accumulated = copy.deepcopy(sensor_data_initial)
        self._opts = opts if opts is not None else ErrGraphOpts()

        if self._opts.force_dependence is not None:
            for node in self._nodes.values():
                node.simulator.set_error_dep(self._opts.force_dependence)

        self._execution_order = self._compile_graph()

        self._errs_systematic = np.zeros(meas_shape)
        self._errs_random = np.zeros(meas_shape)
        self._errs_total = np.zeros(meas_shape)

        self._node_outputs: dict[str, SignalState] | None = (
            {} if self._opts.store_node_outputs else None
        )
        self._node_errors: dict[str, np.ndarray] | None = (
            {} if self._opts.store_node_outputs else None
        )

    def _compile_graph(self) -> tuple[str, ...]:
        """Validates node references and computes topological execution order.
        """
        for name, node in self._nodes.items():
            for inp in node.inputs:
                if inp not in self._nodes:
                    raise KeyError(
                        f"ErrNode '{name}' references non-existent parent "
                        f"input '{inp}'."
                    )

        graph_dict = {
            name: set(node.inputs) for name, node in self._nodes.items()
        }
        try:
            sorter = TopologicalSorter(graph_dict)
            return tuple(sorter.static_order())
        except CycleError as exc:
            raise ValueError(f"Error graph contains a cycle: {exc}") from exc

    def reseed_error_graph(self, seed: int | None = None) -> None:
        """Reseed all random generators in the graph."""
        for node in self._nodes.values():
            node.simulator.reseed(seed)

    def reseed(self, seed: int | None = None) -> None:
        """Alias for reseed_error_graph."""
        self.reseed_error_graph(seed)

    def reseed_error_chain(self, seed: int | None = None) -> None:
        """Compatibility alias for reseed_error_graph."""
        self.reseed_error_graph(seed)

    def calc_errors_from_chain(self, truth: np.ndarray) -> np.ndarray:
        """Compatibility alias for calc_errors_from_graph."""
        return self.calc_errors_from_graph(truth)

    def set_sensor_data_initial(self, sensor_data: SensorData) -> None:
        """Update nominal initial sensor data."""
        self._sens_data_initial = copy.deepcopy(sensor_data)
        self._sens_data_accumulated = copy.deepcopy(sensor_data)

    def calc_errors_from_graph(self, truth: np.ndarray) -> np.ndarray:
        """Calculates total measurement errors by evaluating the graph."""
        self._sens_data_accumulated = copy.deepcopy(self._sens_data_initial)
        self._errs_systematic = np.zeros_like(truth)
        self._errs_random = np.zeros_like(truth)
        self._errs_total = np.zeros_like(truth)

        if self._opts.store_node_outputs:
            self._node_outputs = {}
            self._node_errors = {}

        states: dict[str, SignalState] = {}
        errors: dict[str, np.ndarray] = {}

        root_state = SignalState(
            values=truth,
            sensor_data=copy.deepcopy(self._sens_data_initial),
        )

        for name in self._execution_order:
            node = self._nodes[name]

            # 1. Resolve input signal state
            if not node.inputs:
                in_state = root_state
            elif len(node.inputs) == 1:
                in_state = states[node.inputs[0]]
            else:
                in_state = self._combine_parent_states(
                    [states[p] for p in node.inputs],
                    truth,
                )

            # 2. Evaluate error simulator
            if node.simulator.get_error_dep() == EErrDep.INDEPENDENT:
                err_basis = truth
                err_sens = self._sens_data_initial
            else:
                err_basis = in_state.values
                err_sens = in_state.sensor_data

            error_array, sens_data_perturbed = node.simulator.sim_errs(
                err_basis,
                err_sens,
            )

            # 3. Apply combination operator
            out_state = self._apply_operator(
                node,
                in_state,
                error_array,
                sens_data_perturbed,
            )

            states[name] = out_state
            errors[name] = error_array

            # 4. Accumulate type breakdown
            if node.simulator.get_error_type() == EErrType.SYSTEMATIC:
                self._errs_systematic += error_array
            else:
                self._errs_random += error_array

            if self._opts.store_node_outputs and self._node_outputs is not None:
                self._node_outputs[name] = copy.deepcopy(out_state)
                if self._node_errors is not None:
                    self._node_errors[name] = error_array

        # Resolve terminal state across all leaf nodes
        leaf_names = [
            n
            for n in self._nodes
            if not any(n in other.inputs for other in self._nodes.values())
        ]
        if len(leaf_names) == 1:
            final_state = states[leaf_names[0]]
        elif len(leaf_names) > 1:
            final_state = self._combine_parent_states(
                [states[leaf] for leaf in leaf_names],
                truth,
            )
        else:
            final_state = root_state

        self._errs_total = final_state.values - truth
        self._sens_data_accumulated = final_state.sensor_data

        return self._errs_total

    def _combine_parent_states(
        self,
        parents: Sequence[SignalState],
        truth: np.ndarray,
    ) -> SignalState:
        """Combines multiple parent signal states into an aggregate input
        state.
        """
        # Sum parent deviations from ground truth
        combined_values = truth.copy()
        for p in parents:
            combined_values += (p.values - truth)

        # Merge sensor perturbations: take latest non-nominal state
        combined_sens = copy.deepcopy(parents[-1].sensor_data)
        return SignalState(values=combined_values, sensor_data=combined_sens)

    def _apply_operator(
        self,
        node: ErrNode,
        in_state: SignalState,
        error_array: np.ndarray,
        sens_perturbed: SensorData,
    ) -> SignalState:
        """Applies node operator to generate the output signal state."""
        if node.op == EErrOp.ADD:
            out_values = in_state.values + error_array
            out_sens = copy.deepcopy(sens_perturbed)
            return SignalState(values=out_values, sensor_data=out_sens)

        if node.op == EErrOp.MULTIPLY:
            out_values = in_state.values * (1.0 + error_array)
            out_sens = copy.deepcopy(sens_perturbed)
            return SignalState(values=out_values, sensor_data=out_sens)

        if node.op == EErrOp.REPLACE:
            out_values = error_array.copy()
            out_sens = copy.deepcopy(sens_perturbed)
            return SignalState(values=out_values, sensor_data=out_sens)

        if node.op == EErrOp.CUSTOM:
            if node.custom_op is None:
                raise ValueError(
                    f"ErrNode '{node.name}' has op=EErrOp.CUSTOM but no "
                    f"custom_op provided."
                )
            return node.custom_op(in_state, error_array, sens_perturbed)

        raise ValueError(f"Unsupported EErrOp: {node.op}")

    @property
    def nodes(self) -> dict[str, ErrNode]:
        """Dictionary of node identifiers to ErrNode instances."""
        return self._nodes

    @property
    def execution_order(self) -> tuple[str, ...]:
        """Topological execution order of node names."""
        return self._execution_order

    def get_execution_order(self) -> tuple[str, ...]:
        """Topological execution order of node names."""
        return self._execution_order

    def get_errs_systematic(self) -> np.ndarray:
        """Gets the total systematic error array."""
        return self._errs_systematic

    def get_errs_random(self) -> np.ndarray:
        """Gets the total random error array."""
        return self._errs_random

    def get_errs_total(self) -> np.ndarray:
        """Gets the total measurement error array."""
        return self._errs_total

    def get_sens_data_accumulated(self) -> SensorData:
        """Gets the final accumulated sensor parameters."""
        return self._sens_data_accumulated

    def get_node_outputs(self) -> dict[str, SignalState] | None:
        """Gets intermediate signal states if store_node_outputs=True."""
        return self._node_outputs

    def get_node_errors(self) -> dict[str, np.ndarray] | None:
        """Gets intermediate raw error arrays if store_node_outputs=True."""
        return self._node_errors


class ErrGraphBuilder:
    """Fluent builder for constructing an ErrGraph."""

    __slots__ = ("_nodes",)

    def __init__(self) -> None:
        self._nodes: list[ErrNode] = []

    def add_root(
        self,
        name: str,
        simulator: IErrSimulator,
        op: EErrOp = EErrOp.ADD,
        custom_op: (
            Callable[[SignalState, np.ndarray, SensorData], SignalState] | None
        ) = None,
    ) -> ErrGraphBuilder:
        """Add a root node taking the ground truth input."""
        self._nodes.append(
            ErrNode(
                name=name,
                simulator=simulator,
                inputs=(),
                op=op,
                custom_op=custom_op,
            )
        )
        return self

    def add_child(
        self,
        name: str,
        simulator: IErrSimulator,
        parent: str,
        op: EErrOp = EErrOp.ADD,
        custom_op: (
            Callable[[SignalState, np.ndarray, SensorData], SignalState] | None
        ) = None,
    ) -> ErrGraphBuilder:
        """Add a child node connected to a single parent."""
        self._nodes.append(
            ErrNode(
                name=name,
                simulator=simulator,
                inputs=(parent,),
                op=op,
                custom_op=custom_op,
            )
        )
        return self

    def add_node(
        self,
        name: str,
        simulator: IErrSimulator,
        inputs: Sequence[str],
        op: EErrOp = EErrOp.ADD,
        custom_op: (
            Callable[[SignalState, np.ndarray, SensorData], SignalState] | None
        ) = None,
    ) -> ErrGraphBuilder:
        """Add a node connected to arbitrary parent inputs."""
        self._nodes.append(
            ErrNode(
                name=name,
                simulator=simulator,
                inputs=tuple(inputs),
                op=op,
                custom_op=custom_op,
            )
        )
        return self

    def build(
        self,
        meas_shape: tuple[int, int, int],
        sensor_data_initial: SensorData,
        opts: ErrGraphOpts | None = None,
    ) -> ErrGraph:
        """Build the configured ErrGraph instance."""
        return ErrGraph(
            nodes=self._nodes,
            meas_shape=meas_shape,
            sensor_data_initial=sensor_data_initial,
            opts=opts,
        )


def err_chain_to_graph(
    err_chain: Sequence[IErrSimulator],
    meas_shape: tuple[int, int, int],
    sensor_data_initial: SensorData,
    opts: ErrGraphOpts | None = None,
) -> ErrGraph:
    """Converts a sequential list of error simulators into an ErrGraph.

    Maintains exact dependency semantics: dependent errors connect to the
    preceding node, while independent errors connect to root (ground truth).
    """
    nodes: list[ErrNode] = []
    prev_node_name: str | None = None

    for ii, sim in enumerate(err_chain):
        node_name = f"node_{ii}_{sim.__class__.__name__}"
        parent_inputs = () if prev_node_name is None else (prev_node_name,)

        nodes.append(
            ErrNode(
                name=node_name,
                simulator=sim,
                inputs=parent_inputs,
                op=EErrOp.ADD,
            )
        )
        prev_node_name = node_name

    graph_opts = opts if opts is not None else ErrGraphOpts()
    return ErrGraph(
        nodes=nodes,
        meas_shape=meas_shape,
        sensor_data_initial=sensor_data_initial,
        opts=graph_opts,
    )
