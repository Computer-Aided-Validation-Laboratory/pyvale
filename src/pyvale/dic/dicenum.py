# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from enum import Enum


"""
Enums for the DIC Engine
"""

class EScanMethod(str, Enum):
    MULTIWINDOW_RG = "MULTIWINDOW_RG"
    SINGLEWINDOW_RG = "SINGLEWINDOW_RG"
    MULTIWINDOW = "MULTIWINDOW"
    RASTER = "RASTER"

class EShape(str, Enum):
    RIGID = "RIGID"
    AFFINE = "AFFINE"
    QUAD = "QUAD"

class ECorrCrit(str, Enum):
    SSD = "SSD"
    NSSD = "NSSD"
    ZNSSD = "ZNSSD"

class EInterp(str, Enum):
    BSPLINE = "BSPLINE"
    HERMITE = "HERMITE"

class EIncrementalMethod(str, Enum):
    OFF = "OFF"
    IMAGE = "IMAGE"
    COST = "COST"
    ITER = "ITER"
