"""
===============================================================================
OutpuReader Abstract Base Class

Authors: Lloyd Fletcher
===============================================================================
"""
from abc import ABC, abstractmethod
from pathlib import Path
from mooseherder.simdata import SimData, SimReadConfig


class OutputReader(ABC):
    """ """

    @abstractmethod
    def __init__(self, output_file: Path) -> None:
        pass

    @abstractmethod
    def read_sim_data(self, read_config: SimReadConfig) -> SimData:
        """

        Parameters
        ----------
        read_config: SimReadConfig :
            

        Returns
        -------

        """
        pass

    @abstractmethod
    def read_all_sim_data(self) -> SimData:
        """ """
        pass
