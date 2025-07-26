"""
===============================================================================
SimRunner Abstract Base Class

Authors: Lloyd Fletcher
===============================================================================
"""
from abc import ABC, abstractmethod
from pathlib import Path

class SimRunner(ABC):
    """SimRunner: ABC for the moosherd simulation chain. A simulation has an
    input file which can be get or set. The simulation can then be run with the
    given input files and then the path to the simulation output file can be
    retrieved.

    Parameters
    ----------

    Returns
    -------

    """
    @abstractmethod
    def get_input_file(self) -> Path | None:
        """get_input_path"""


    @abstractmethod
    def set_input_file(self, input_path: Path) -> None:
        """set_input_file

        Parameters
        ----------
        input_path: Path :
            

        Returns
        -------

        """


    @abstractmethod
    def run(self, input_file: Path | None = None) -> None:
        """run

        Parameters
        ----------
        input_file: Path | None :
             (Default value = None)

        Returns
        -------

        """

    @abstractmethod
    def get_output_path(self) -> Path | None:
        """get_output_path"""

