# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ==============================================================================

from dataclasses import dataclass, field
import numpy as np


@dataclass(slots=True)
class ExpData:
    # shape=(n_sensors,n_time_steps)
    fields: np.ndarray

    # Use these to index into n_sensors axis
    sens_label_to_ind: dict[str,int]
    ind_to_sens_label: dict[int,str]

    # shape=(n_sensors,3) where 3=coord[x,y,z]
    coords: np.ndarray | None = None
    # shape=(n_time_steps,)
    times: np.ndarray | None = None
    
    
