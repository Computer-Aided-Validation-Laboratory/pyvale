# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Common loading and resource-lifetime interface for simulation output files."""

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from pyvale.dataio.simdata import SimData, SimLoadConfig


class IOutputLoader(ABC):
    """Load simulation output and explicitly release its backing resources.

    Notes
    -----
    Subclasses implement the loading methods and an idempotent ``close``.
    The inherited context manager closes the loader on normal exit and on
    exceptions, without suppressing an exception raised inside the block.
    Loaded arrays must remain usable after the backing resource is closed.
    """

    def __enter__(self) -> Self:
        """Return this loader when entering a ``with`` block.

        Returns
        -------
        Self
            The concrete loader instance, preserving its subclass type.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the loader when leaving a ``with`` block.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type raised by the block, or None on normal exit.
        exc_value : BaseException | None
            Exception raised by the block, or None on normal exit.
        traceback : TracebackType | None
            Traceback associated with the exception, or None on normal exit.

        Notes
        -----
        Returning None leaves exceptions unsuppressed. Cleanup exceptions also
        propagate; resources are not left to garbage collection for closure.
        """
        self.close()

    @abstractmethod
    def close(self) -> None:
        """Release the backing resource, if it is still open.

        Notes
        -----
        Implementations must allow repeated calls, including an explicit close
        inside a ``with`` block followed by the automatic close on exit.
        """

    @abstractmethod
    def load_sim_data(self, read_config: SimLoadConfig) -> SimData:
        """Load selected simulation arrays into memory.

        Parameters
        ----------
        read_config : SimLoadConfig
            Fields, geometry, connectivity and time indices to load.

        Returns
        -------
        SimData
            Selected simulation data, independent of the loader's lifetime.
        """

    @abstractmethod
    def load_all_sim_data(self) -> SimData:
        """Load all available simulation arrays into memory.

        Returns
        -------
        SimData
            Available simulation data, independent of the loader's lifetime.
        """
