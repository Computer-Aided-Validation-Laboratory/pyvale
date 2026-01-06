#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import pytest
from pyvale.mooseherder.sweeploader import SweepLoader
from pyvale.mooseherder.directorymanager import DirectoryManager
import tests.mooseherder.herdchecker as hc


@pytest.fixture
def dir_manager() -> DirectoryManager:
    this_manager = DirectoryManager(hc.NUM_DIRS)
    this_manager.set_base_dir(hc.OUTPUT_GOLD_PATH)
    return this_manager


@pytest.fixture
def sweep_reader(dir_manager) -> SweepLoader:
    return SweepLoader(dir_manager,hc.NUM_PARA)


def test_init_sweep_reader(sweep_reader: SweepLoader) -> None:
    assert sweep_reader is not None
    assert sweep_reader._dir_manager is not None
    assert sweep_reader._n_para_read == hc.NUM_PARA
    assert len(sweep_reader._output_files) == 0

