# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ==============================================================================

from dataclasses import dataclass, field
import numpy as np

@dataclass(slots=True)
class ExpData:
    fields: dict[str,np.ndarray] = field(default_factory=dict)
    coords: dict[str,np.ndarray] = field(default_factory=dict)
    times: dict[str,np.ndarray] = field(default_factory=dict)
    
