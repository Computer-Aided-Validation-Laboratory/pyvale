# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
from abc import ABC, abstractmethod
from pathlib import Path
from pyvale.mooseherder.simdata import SimData, SimReadConfig


class OutputReader(ABC):
    @abstractmethod
    def __init__(self, output_file: Path) -> None:
        pass

    @abstractmethod
    def read_sim_data(self, read_config: SimReadConfig) -> SimData:
        pass

    @abstractmethod
    def read_all_sim_data(self) -> SimData:
        pass
