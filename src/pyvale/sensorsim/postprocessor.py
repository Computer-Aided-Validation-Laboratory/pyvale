# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Base interface and composition pipelines for post-measurement signal
processing.
"""

from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

from pyvale.sensorsim.measurementdata import MeasurementData


class IMeasurementProcessor(ABC):
    """Abstract base interface for all experimental post-measurement signal
    processors and derived quantity calculators.
    """

    __slots__ = ()

    @abstractmethod
    def get_source_keys(self) -> tuple[str, ...]:
        """Returns the names of required input sensor or measurement keys."""
        pass

    @abstractmethod
    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        """Returns the names of output components produced by this processor."""
        pass

    @abstractmethod
    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        """Executes post-processing on the input measurement datasets.

        Parameters
        ----------
        inputs : dict[str, MeasurementData]
            Dictionary of input measurement datasets mapped by their source key.

        Returns
        -------
        MeasurementData
            Derived or processed measurement data object.
        """
        pass


class ProcessingPipeline(IMeasurementProcessor):
    """Executes a sequential linear chain of single-input signal processors.
    """

    __slots__ = ("_source_key", "_steps")

    def __init__(
        self,
        source: str,
        steps: Sequence[IMeasurementProcessor],
    ) -> None:
        """
        Parameters
        ----------
        source : str
            Identifier of the initial input measurement dataset.
        steps : Sequence[IMeasurementProcessor]
            Sequential list of processing steps to evaluate in order.
        """
        self._source_key = source
        self._steps = tuple(steps)

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source_key,)

    def get_steps(self) -> tuple[IMeasurementProcessor, ...]:
        return self._steps

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        current_meta = input_metadata
        for step in self._steps:
            current_src = step.get_source_keys()
            src_name = current_src[0] if current_src else self._source_key
            out_comps = step.get_output_components(current_meta)
            first_meta = next(iter(current_meta.values()))
            ref_data = current_meta.get(src_name, first_meta)
            dummy_data = MeasurementData(
                values=np.empty((1, len(out_comps), 1)),
                sample_times=ref_data.sample_times,
                positions=ref_data.positions,
                components=out_comps,
                units=ref_data.units,
                descriptor=ref_data.descriptor,
            )
            current_meta = {src_name: dummy_data}
        return current_meta[src_name].components

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        if self._source_key not in inputs:
            raise KeyError(
                f"ProcessingPipeline requires source '{self._source_key}', "
                f"but inputs provided: {list(inputs.keys())}"
            )
        current_data = inputs[self._source_key]
        for step in self._steps:
            step_sources = step.get_source_keys()
            src_name = step_sources[0] if step_sources else self._source_key
            current_data = step.process({src_name: current_data})
        return current_data
