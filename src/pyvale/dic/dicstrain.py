# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import matplotlib.pyplot as plt
import numpy as np

from pyvale.dic import dic2dcpp

def DICstrain(dic_data, 
                         window_size: int=5, 
                         window_element: int=4, 
                         strain_formulation: str="HENCKY"):


    # Check the strain formulation is in the allowed list
    allowed_formulations = ["GREEN", "ALMANSI", "HENCKY", 
                            "BIOT_EULER", "BIOT_LAGRANGE"]
    
    if strain_formulation not in allowed_formulations:
        raise ValueError(f"Invalid strain formulation: "
                         f"'{strain_formulation}'. Allowed values are: "
                         f"{', '.join(allowed_formulations)}.")

    # check the strain window element is one of the allowed values
    allowed_element = [4, 9]
    if window_element not in allowed_element:
        raise ValueError(f"Invalid strain window element type: "
                         f"Q{window_element}. Allowed values are: "
                         f"{', '.join(map(str, allowed_element))}.")

    # chceck the window size is an odd number
    if window_size % 2 == 0:
        raise ValueError(f"Invalid strain window size: '{window_size}'. "
                         f"Must be an odd number.")

    # convert subset data to meshgrid
    x, y = dic_data[:, 0], dic_data[:, 1]

    x_unique = np.unique(x).astype(np.int32)
    y_unique = np.unique(y).astype(np.int32)

    x, y = np.meshgrid(x_unique, y_unique)

    u_mesh = np.full_like(x, np.nan, dtype=np.float64)
    v_mesh = np.full_like(y, np.nan, dtype=np.float64)

    for i in range(len(dic_data)):
        # Find indices in meshgrid
        xi, yi, ui, vi = dic_data[i]
        x_idx = np.where(x_unique == xi)[0][0]
        y_idx = np.where(y_unique == yi)[0][0]
        u_mesh[y_idx, x_idx] = ui
        v_mesh[y_idx, x_idx] = vi

    plt.plot()
    plt.pcolor(x, y, u_mesh)
    plt.colorbar()
    plt.show()
    dudx = np.gradient(u_mesh, axis=0)
    dudy = np.gradient(u_mesh, axis=1)
    dvdx = np.gradient(v_mesh, axis=0)
    dvdy = np.gradient(v_mesh, axis=1)

    np.savetxt("dudx.dat",dudx,delimiter=" ")
    np.savetxt("dudy.dat",dudy,delimiter=" ")
    np.savetxt("dvdx.dat",dvdx,delimiter=" ")
    np.savetxt("dvdy.dat",dvdy,delimiter=" ")


    dic2dcpp.cpp_2d_strain_routine(x,y,u_mesh,v_mesh, window_size,
                                          window_element, strain_formulation)



