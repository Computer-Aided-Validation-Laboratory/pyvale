# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
from abc import ABC, abstractmethod
from pyvale.mooseherder.simdata import SimData, SimLoadConfig


class IOutputLoader(ABC):
    @abstractmethod
    def load_sim_data(self, read_config: SimLoadConfig) -> SimData:
        pass

    @abstractmethod
    def load_all_sim_data(self) -> SimData:
        pass
