# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Analytic plate with a fast heating pulse and a smooth decay."""

from dataclasses import replace
from typing import cast

import numpy as np
import numpy.typing as npt

import pyvale.dataio as io
from pyvale.verif.analyticsimdatafactory import scalar_linear_2d


def build_thermal_pulse() -> io.SimData:
    """Build a heating blob that rises to its peak at 0.45 s."""
    plate, _ = scalar_linear_2d()
    coords = cast(npt.NDArray[np.float64], plate.coords)
    x, y = coords[:, 0], coords[:, 1]
    times = np.linspace(0.0, 1.0, 41)
    peak_time = 0.45
    envelope = np.exp(-0.5 * ((times - peak_time) / 0.18) ** 2)
    ramp = times < peak_time
    envelope[ramp] = np.clip((times[ramp] - 0.35) / (peak_time - 0.35), 0.0, 1.0)
    blob = np.exp(-0.5 * (((x - 3.75) / 1.2) ** 2 + ((y - 4.5) / 1.0) ** 2))
    temperature = 20.0 + 18.0 * blob[:, None] * envelope[None, :]
    return replace(plate, time=times, node_vars={"temperature": temperature})
