# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Directed Acyclic Graph (DAG) orchestration engine for post-measurement
signal processing.
"""

from graphlib import CycleError, TopologicalSorter
import numpy as np

from pyvale.sensorsim.measurementdata import MeasurementData
from pyvale.sensorsim.postprocessor import (
    IMeasurementProcessor,
    ProcessingPipeline,
)


class PostProcessGraph:
    """Orchestrates post-measurement processing pipelines and multi-sensor
    derived quantity calculators using a directed acyclic graph.
    """

    __slots__ = ("_processors", "_topo_order")

    def __init__(
        self,
        processors: dict[str, IMeasurementProcessor | ProcessingPipeline]
        | None = None,
    ) -> None:
        self._processors: dict[
            str, IMeasurementProcessor | ProcessingPipeline
        ] = {}
        self._topo_order: tuple[str, ...] = ()

        if processors:
            for name, proc in processors.items():
                self.add_processor(name, proc)

    def add_processor(
        self,
        name: str,
        processor: IMeasurementProcessor | ProcessingPipeline,
    ) -> "PostProcessGraph":
        """Adds a named processing step or pipeline to the graph."""
        self._processors[name] = processor
        self._compile()
        return self

    def _compile(self) -> None:
        dep_graph: dict[str, set[str]] = {}
        for name, proc in self._processors.items():
            dep_graph[name] = set(proc.get_source_keys())

        # Only include internal dependency edges between processors in graph
        sorter_deps: dict[str, set[str]] = {}
        for name, sources in dep_graph.items():
            internal_sources = {
                src for src in sources if src in self._processors
            }
            sorter_deps[name] = internal_sources

        sorter = TopologicalSorter(sorter_deps)
        try:
            order = tuple(sorter.static_order())
        except CycleError as exc:
            raise ValueError(
                f"Cyclic dependency detected in PostProcessGraph: {exc}"
            ) from exc

        self._topo_order = order

    def get_processor(
        self,
        name: str,
    ) -> IMeasurementProcessor | ProcessingPipeline:
        return self._processors[name]

    def get_all_processor_names(self) -> tuple[str, ...]:
        return tuple(self._processors.keys())

    def get_execution_order(self) -> tuple[str, ...]:
        return self._topo_order

    def execute(
        self,
        raw_measurements: dict[str, MeasurementData | np.ndarray],
    ) -> dict[str, MeasurementData]:
        """Executes the post-processing DAG over input measurement datasets.

        Parameters
        ----------
        raw_measurements : dict[str, MeasurementData | np.ndarray]
            Dictionary of base sensor measurement datasets.

        Returns
        -------
        dict[str, MeasurementData]
            Dictionary containing all processed and derived measurement
            datasets.
        """
        all_data: dict[str, MeasurementData] = {}

        # Convert raw arrays into MeasurementData if needed
        for k, v in raw_measurements.items():
            if isinstance(v, MeasurementData):
                all_data[k] = v
            else:
                arr = np.asarray(v)
                n_times = arr.shape[-1]
                t_arr = np.linspace(0.0, 1.0, n_times)
                all_data[k] = MeasurementData(
                    values=arr,
                    sample_times=t_arr,
                    positions=None,
                    components=("value",),
                    units="",
                )

        for name in self._topo_order:
            proc = self._processors[name]
            needed_keys = proc.get_source_keys()
            inputs_for_proc: dict[str, MeasurementData] = {}
            for k in needed_keys:
                if k not in all_data:
                    raise KeyError(
                        f"Processor '{name}' requires missing source '{k}'"
                    )
                inputs_for_proc[k] = all_data[k]

            all_data[name] = proc.process(inputs_for_proc)

        return all_data
