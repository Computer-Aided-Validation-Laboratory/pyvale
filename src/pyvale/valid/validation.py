# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

from pyvale.dataio.expdata import ExpData
from pyvale.sensorsim.generatorsrandom import IGenRandom

#-------------------------------------------------------------------------------
# Data Structures

# @dataclass(slots=True)
# class ValData:
#     val_data: np.ndarray
#     epistemic_intervals: np.ndarray | None = None

TIME_IND: int = -1
SENS_IND: int = 0 
EPIS_IND: int = 1
ALEA_IND: int = 2

@dataclass(slots=True)
class PointValData:
    val_points: dict[str,np.ndarray] = field(default_factory=dict)
    """SIM:shape=(n_sensors,n_epistemic,n_aleatory)
       EXP:shape=(n_sensors,n_epistemic,n_steady_repeats)
    """

    epistemic_intervals: dict[str,np.ndarray | None] = field(default_factory=dict)
    """shape=(n_sensors,2), where 2 = (low,high)
    """

    val_label_to_ind: dict[tuple[str,str],int] = field(default_factory=dict)
    ind_to_val_label: dict[tuple[str,int],str] = field(default_factory=dict)
    """Use these to index into the above numpy arrays
    """
    
    #TODO
    #coords
    #time    

# TODO: 
# - Allow time slicing here as well as on exp load, here slicing is for steady 
#   state.
# 
def extract_val_data_by_key(
    exp_data: ExpData,
    epistemic_intervals: dict[str,np.ndarray | None],
    sensor_keys: dict[str,list[str] | None],
    steady_slice: dict[str,slice | None] | None = None,
) -> PointValData:

    # 1. If sensor_keys 
    val_data = PointValData()
    for array_key,sens_list in sensor_keys.items():
        
        if sens_list is None:
            val_data.val_points[array_key] = exp_data.fields[array_key] 
            continue

        # Allocate a numpy array based on how many sensors we want to extract
        # and analyse

        #TODO: extract the val_points for each sensor list here.

def extract_val_data_by_slice(
    exp_data: ExpData,
    epistemic_intervals: dict[str,np.ndarray | None],
    sensor_keys: dict[str,np.ndarray | slice],
    steady_slice: dict[str,slice | None] | None = None,
) -> PointValData:        


    return val_data

@dataclass(slots=True)
class ImageValData:
    val_images: np.ndarray
    epistemic_intervals: np.ndarray | None = None

ValData = PointValData | ImageValData 

#-------------------------------------------------------------------------------
# IO and synthetic data generation

def load_val_data(load_file: Path) -> ValData:
    pass

def gen_val_data(nominal_data: np.ndarray,
                 aleatory_gen: IGenRandom | None, 
                 epistemic_gen: IGenRandom| None) -> ValData:
    pass
    
#-------------------------------------------------------------------------------
# Data Analysis

# TODO: ECDF limit calculation function
def calc_limit_cdfs_point(val_data: PointValData
                          ) -> dict[tuple[str,...],np.ndarray]:
    pass
     
# TODO: MAVM calculation function
def calc_mavm_point(exp_data: PointValData,
                    sim_data: PointValData, 
                    ) -> dict[tuple[str,...],np.ndarray]:
    pass 

#-------------------------------------------------------------------------------
# Visualisation

    
#-------------------------------------------------------------------------------
# Tools / Helper Functions
def vectorised_ecdf(data: np.ndarray, 
                    axis: int
                    ) -> tuple[np.ndarray,np.ndarray]:
    pass
