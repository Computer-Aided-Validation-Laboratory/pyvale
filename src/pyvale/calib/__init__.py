#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

"""Calibration tools for stereo DIC workflows.

This package exposes the high-level calibration utilities used to detect dot
calibration targets and estimate stereo camera parameters. The main public
entry points are :func:`detect_dots`, :func:`calibrate_stereo`, and the
:class:`Calib` and :class:`CamIntrinsics` result containers.
"""

from .calibdotdetect import detect_dots
from .calibstereo import calibrate_stereo
from .calibdataclass import Calib, CamIntrinsics, savetxt, loadtxt

__all__ = ["detect_dots",
           "Calib",
           "CamIntrinsics",
           "calibrate_stereo",
           "savetxt",
           "loadtxt"]
