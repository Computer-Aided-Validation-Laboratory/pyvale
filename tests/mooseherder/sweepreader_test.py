#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import pytest
from pyvale.mooseherder.sweepreader import SweepReader
from pyvale.mooseherder.directorymanager import DirectoryManager
import tests.mooseherder.herdchecker as hc


@pytest.fixture
def dir_manager() -> DirectoryManager:
    this_manager = DirectoryManager(hc.NUM_DIRS)
    this_manager.set_base_dir(hc.OUTPUT_PATH)
    return this_manager


@pytest.fixture
def sweep_reader(dir_manager) -> SweepReader:
    return SweepReader(dir_manager,hc.NUM_PARA)


def test_init_sweep_reader(sweep_reader: SweepReader) -> None:
    assert sweep_reader is not None
    assert sweep_reader._dir_manager is not None
    assert sweep_reader._n_para_read == hc.NUM_PARA
    assert len(sweep_reader._output_files) == 0

