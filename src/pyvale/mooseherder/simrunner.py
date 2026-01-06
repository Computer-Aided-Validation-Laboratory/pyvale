# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from abc import ABC, abstractmethod
from pathlib import Path

class SimRunner(ABC):
    """SimRunner: ABC for the moosherd simulation chain. A simulation has an
    input file which can be get or set. The simulation can then be run with the
    given input files and then the path to the simulation output file can be
    retrieved.
    """
    @abstractmethod
    def get_input_file(self) -> Path | None:
        pass

    @abstractmethod
    def set_input_file(self, input_path: Path) -> None:
        pass

    @abstractmethod
    def run(self, input_file: Path | None = None) -> None:
        pass

    @abstractmethod
    def get_output_path(self) -> Path | None:
        pass

