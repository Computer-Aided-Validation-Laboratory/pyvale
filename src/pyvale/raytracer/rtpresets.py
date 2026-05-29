# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np
from dataclasses import dataclass, field

# ================================================================================
# MATERIAL PRESETS
# ================================================================================

@dataclass(slots=True)
class Material:
    """
    Convenience dataclass for storing material data.

    Parameters:
    -----------
    color: np.ndarray
        A (3,1) NumPy array storing the material colour data. For refractive materials, it is the tint at the specified reference thickness.
    RI: float | None
        The refractive index of the material. Defaults to None.
    """
    color: np.ndarray = field(default_factory= lambda: np.array([1.0,1.0,1.0]))
    RI: float | None = field(default_factory=None)

# Data sources:
# [1] https://physicallybased.info
# [2] Old Mitsuba documentation: https://www.mitsuba-renderer.org/releases/0.4.5/documentation.pdf chapter 8\

# Nb4: for more accurate data, we could also use https://refractiveindex.info/
# but this is when we get to things like wavelengths, etc.

class MaterialPresets:
    """
    Contains material presets for convenient use. MaterialType still needs to be specified manually.
    """
    AIR = Material(np.array([1.000,1.000,1.000]), 1.000277) # [2]
    ALUMINUM = Material(np.array([0.916,0.923,0.924]), None) # [1]
    BRASS = Material(np.array([0.910,0.778,0.423]), None) # [1]
    COPPER = Material(np.array([0.932, 0.623, 0.522]), None) # [1]
    CONCRETE = Material(np.array([0.510,0.510,0.510]), 1.500) # [1]
    DIAMOND = Material(np.array([1.000,1.000,1.000]), 1.348) # [1]
    GLASS_BOROSILICATE = Material(np.array([0.988,0.992,0.985]), 1.520)  # [1]
    GLASS_SODA_LIME = Material(np.array([0.984,0.995,0.995]), 1.520) # Accounts for about 90% of the glass made; [1]
    GOLD = Material(np.array([1.059,0.773,0.307]), None) # [1]
    HONEY_LIQUID = Material(np.array([0.831,0.571,0.037]), 1.504) # [1]
    ICE = Material(np.array([0.973, 0.995, 1.000]), 1.310) # [1]
    IRON = Material(np.array([0.530,0.513,0.494]), None) # [1]
    LEAD = Material(np.array([0.626,0.640,0.693]), None) # [1]
    MUSOU_BLACK = Material(np.array([0.006,0.006,0.006]), 1.500) # The paint, [1]
    OFFICE_PAPER = Material(np.array([0.794,0.834,0.884]), 1.500) # [1]
    PEARL = Material(np.array([0.800,0.750,0.700]), 1.680) # [1]
    PLASTIC_ACRYLIC = Material(np.array([1.000,1.000,1.000]), 1.490) # [1]
    PLASTIC_PC = Material(np.array([1.000,1.000,1.000]), 1.585) # [1]
    PLASTIC_PET = Material(np.array([1.000,1.000,1.000]), 1.575) # [1]
    PLASTIC_PP = Material(np.array([1.000,1.000,1.000]), 1.492) # [1]
    PLASTIC_PUR = Material(np.array([1.000,1.000,1.000]), 1.600) # [1]
    PLASTIC_PVC = Material(np.array([1.000,1.000,1.000]), 1.542) # [1]
    SOAP_BUBBLE = Material(np.array([1.000,1.000,1.000]), 1.000) # [1]
    STAINLESS_STEEL = Material(np.array([0.669,0.639,0.598]), None) # [1]
    TUNGSTEN = Material(np.array([0.537,0.536,0.519]), None) # [1]
    WATER = Material(np.array([0.969,0.996,0.997]), 1.333) # [1]
    