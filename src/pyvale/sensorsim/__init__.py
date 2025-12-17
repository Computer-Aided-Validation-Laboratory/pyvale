# ==============================================================================
# . the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
The python validation engine is your virtual engineering laboratory:
An all-in-one package for sensor simulation, sensor uncertainty quantification,
sensor placement optimisation and simulation calibration/validation. Used to
simulate experimental data from an input multi-physics simulation by explicitly
modelling sensors with realistic uncertainties. Useful for experimental design,
sensor placement optimisation, testing simulation validation metrics and
virtually testing digital shadows/twins.
"""

# NOTE: this simplifies and decouples how the user calls .from the
# underlying project structure: the user should be able to use '.'
# and access everything in one layer without multiple import dots

from .field import *
from .fieldscalar import *
from .fieldvector import *
from .fieldtensor import *
from .fieldconverter import *
from .fieldtransform import *
from .fieldinterp import *
from .fieldinterpmesh import *
from .fieldinterppoints import*

from .integratorspatial import *
from .integratorquadrature import *
from .integratorrectangle import *
from .integratorfactory import *

from .sensordescriptor import *
from .sensortools import *
from .sensorarray import *
from .sensorfactory import *
from .sensorspoint import *
from .sensordata import *

from .camera import *
from .cameradata import *
from .cameradata2d import *
from .cameratools import *
from .camerastereo import *

from .rastercy import *

from .renderscene import *
from .rendermesh import *
from .rasternp import *

from .imagedef2d import *

from .errorintegrator import *
from .errorrand import *
from .errorsysindep import *
from .errorsysdep import *
from .errorsysfield import *
from .errorsyscalib import *
from .errordriftcalc import *

from .generatorsrandom import *

from .visualopts import *
from .visualtools import *
from .visualsimsensors import *
from .visualsimanimator import *
from .visualexpplotter import *
from .visualtraceplotter import *
from .visualimages import *
from .visualimagedef import *
from .visualtraceanimator import *

from .simtools import *

from .experimentsimulator import *
from .experimentstats import *
from .experimentsimio import *

from .enums import *
from .exceptions import *


