# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
`pyvale`: the python validation engine. Used to simulate experimental data from
an input multi-physics simulation by explicitly modelling sensors with realistic
uncertainties. Useful for experimental design, sensor placement optimisation,
testing simulation validation metrics and testing digital shadows/twins.
"""

# NOTE: this simplifies and decouples how the user calls pyvale from the
# underlying project structure: the user should be able to use 'pyvale.'
# and access everything in one layer without multiple import dots


from pyvale.dataset import *

from pyvale.field import *
from pyvale.fieldscalar import *
from pyvale.fieldvector import *
from pyvale.fieldtensor import *
from pyvale.fieldconverter import *
from pyvale.fieldtransform import *

from pyvale.integratorspatial import *
from pyvale.integratorquadrature import *
from pyvale.integratorrectangle import *
from pyvale.integratorfactory import *

from pyvale.sensordescriptor import *
from pyvale.sensortools import *
from pyvale.sensorarray import *
from pyvale.sensorarrayfactory import *
from pyvale.sensorarraypoint import *
from pyvale.sensordata import *

from pyvale.camera import *
from pyvale.cameradata import *
from pyvale.cameradata2d import *
from pyvale.cameratools import *
from pyvale.camerastereo import *

import pyvale.cython.rastercyth as rastercyth
from pyvale.rastercy import *

from pyvale.rendermesh import *
from pyvale.rasternp import *

from pyvale.imagedef2d import *

from pyvale.errorintegrator import *
from pyvale.errorrand import *
from pyvale.errorsysindep import *
from pyvale.errorsysdep import *
from pyvale.errorsysfield import *
from pyvale.errorsyscalib import *
from pyvale.errordriftcalc import *

from pyvale.generatorsrandom import *

from pyvale.visualopts import *
from pyvale.visualtools import *
from pyvale.visualsimsensors import *
from pyvale.visualsimanimator import *
from pyvale.visualexpplotter import *
from pyvale.visualtraceplotter import *
from pyvale.visualimages import *
from pyvale.visualimagedef import *

from pyvale.analyticmeshgen import *
from pyvale.analyticsimdatagenerator import *
from pyvale.analyticsimdatafactory import *

from pyvale.experimentsimulator import *

from pyvale.blendercalibrationdata import *
from pyvale.blenderlightdata import *
from pyvale.blendermaterialdata import *
from pyvale.blenderrenderdata import *
from pyvale.blenderscene import *
from pyvale.blendertools import *
from pyvale.simtools import *

from pyvale.output import *
from pyvale.pyvaleexceptions import *

