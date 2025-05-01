# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

from pyvale.core.dic.python import diccppinterface


def dic_import_data(path, prefix: str="results", format: str=".dat", delim: str=" "):
    """
    Import data from a file. The format of the file is determined by the extension.
    Only reads coords, u, and v values.
    
    Args:
        path (str): Path to the file.
        format (str): Format of the file. Default is ".bin".
    
    Returns:
        data: Imported data containing coords, u, and v values.
    """
    
    if format == ".bin":
        
        # Calculate the size of a row based on known data types
        row_size = (3*4 + 12*8)  # 2 integers (coords) + 12 doubles
        data_list = []

        with open(path, "rb") as f:
            
            while True:
            
                bytes_read = f.read(row_size)
                
                # check the length of the line is what it should be
                if not bytes_read:
                    break
                if len(bytes_read) != row_size:
                    raise ValueError("Incomplete row in binary file.")

                # currently only interested in the coordinates and displacement
                coords = np.frombuffer(bytes_read[:8], dtype=np.int32)
                u_v = np.frombuffer(bytes_read[8:24], dtype=np.float64)

                # Combine coords, u, and v into one row
                row = np.concatenate([coords, u_v])
                data_list.append(row)
        
        # convert list to numpy array
        data = np.array(data_list)

    elif format == ".dat":
        data = np.loadtxt(path, delimiter=delim, 
                          skiprows=0, usecols=(0, 1, 2, 3))

    else:
        raise ValueError(f"Unsupported file format: {format}")

    return data



def dic_calculate_strain(dic_data, 
                         window_size: int=5, 
                         window_element: int=4, 
                         strain_formulation: str="HENCKY"):

    print(dic_data.shape)

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

    # Check the strain formulation is in the allowed list
    allowed_formulations = ["GREEN", "ALMANSI", "HENCKY", 
                            "BIOT_EULER", "BIOT_LAGRANGE"]
    
    if strain_formulation not in allowed_formulations:
        raise ValueError(f"Invalid strain formulation: '{strain_formulation}'. "
                         f"Allowed values are: {', '.join(allowed_formulations)}.")

    # check the strain window element is one of the allowed values
    allowed_element = [4, 9]
    if window_element not in allowed_element:
        raise ValueError(f"Invalid strain window element type: Q{window_element}. "
                         f"Allowed values are: {', '.join(map(str, allowed_element))}.")

    # chceck the window size is an odd number
    if window_size % 2 == 0:
        raise ValueError(f"Invalid strain window size: '{window_size}'. "
                         f"Must be an odd number.")


    diccppinterface.cpp_2d_strain_routine(x,y,u_mesh,v_mesh, window_size,
                                          window_element, strain_formulation)



