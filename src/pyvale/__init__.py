# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
The python validation engine (`pyvale`) is your virtual engineering laboratory:
An all-in-one package for sensor simulation, sensor uncertainty quantification,
sensor placement optimisation and simulation calibration/validation. Used to
simulate experimental data from an input multi-physics simulation by explicitly
modelling sensors with realistic uncertainties. Useful for experimental design,
sensor placement optimisation, testing simulation validation metrics and
virtually testing digital shadows/twins.
"""

from . import dic
from . import strain
from . import blender
from . import sensorsim
from . import mooseherder
from . import dataset
from . import calib
