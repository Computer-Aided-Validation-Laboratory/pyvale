# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Resource ownership and exception behaviour of output-loader contexts."""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

import pyvale.dataio as io
import pyvale.mooseherder as mh


@pytest.mark.parametrize("fail_in_block", [False, True])
def test_exodus_context_closes_and_preserves_loaded_data(fail_in_block: bool) -> None:
    """A real NetCDF file closes on both paths; loaded arrays survive closure."""
    path = Path(__file__).parent / "output_gold/moose-mech-outtest_out.e"
    loader = mh.ExodusLoader(path)
    with (
        pytest.raises(ValueError, match="body failed")
        if fail_in_block
        else nullcontext(),
        loader as reader,
    ):
        assert reader is loader
        data: io.SimData = reader.load_all_sim_data()
        assert reader._data.isopen()
        if fail_in_block:
            raise ValueError("body failed")
    assert not loader._data.isopen()
    assert data.coords.size > 0
    assert np.all(np.isfinite(data.coords))
    loader.close()


def test_close_is_compulsory_for_subclasses() -> None:
    """A loader cannot opt out of the resource-lifetime contract."""

    class MissingClose(mh.IOutputLoader):
        def load_sim_data(self, read_config: io.SimLoadConfig) -> io.SimData:
            return io.SimData()

        def load_all_sim_data(self) -> io.SimData:
            return io.SimData()

    with pytest.raises(TypeError, match="close"):
        MissingClose()


def test_explicit_close_inside_context_is_safe() -> None:
    """Explicit closure followed by context exit must not close NetCDF twice."""
    path = Path(__file__).parent / "output_gold/moose-mech-outtest_out.e"
    with mh.ExodusLoader(path) as reader:
        reader.close()
        assert not reader._data.isopen()
