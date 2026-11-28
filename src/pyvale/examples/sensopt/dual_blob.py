# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Shared analytic field and plotting for the dual-blob placement examples."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pyvista as pv

import pyvale.dataio as io
import pyvale.sensoropt as opt
import pyvale.sensorsim as sens


def build_dual_blob_case(
    mesh: io.SimData,
    *,
    peak: tuple[float, float],
    trough: tuple[float, float],
) -> io.SimData:
    """Put a warm peak and a cool trough on the same plate."""
    coords = cast(npt.NDArray[np.float64], mesh.coords)
    times = cast(npt.NDArray[np.float64], mesh.time)
    x, y = coords[:, 0], coords[:, 1]
    warm = np.exp(-0.5 * (((x - peak[0]) / 0.9) ** 2 + ((y - peak[1]) / 0.8) ** 2))
    cool = np.exp(-0.5 * (((x - trough[0]) / 1.2) ** 2 + ((y - trough[1]) / 0.9) ** 2))
    temperature = 20.0 + (14.0 * warm - 9.0 * cool)[:, None] * (0.5 + 0.5 * times)
    return replace(mesh, node_vars={"temperature": temperature})


def save_dual_blob_layout(
    result: opt.OptResult, array: sens.SensorsPoint, path: Path, *, title: str
) -> None:
    """Overlay selected sites on the nominal temperature field."""
    selected = sens.SensorsPoint(
        result.sensor_data["temperature"], array.get_field(), array.get_descriptor()
    )
    plotter: pv.Plotter = sens.plot_point_sensors_on_sim(
        selected,
        "temperature",
        vis_opts=sens.VisOptsSimSensors(interactive=False, show_edges=False),
    )
    plotter.add_text(title, position="upper_right", font_size=12)
    plotter.screenshot(path)
    plotter.close()
