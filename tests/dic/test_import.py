"""
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
license: mit
copyright (c) 2024 the computer aided validation team
================================================================================
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pyvale.dic as dic

test_dir = os.path.dirname(__file__)


def test_dic_data_column_import():

    filename = "./reference/ref_00_50.csv"
    datafile = os.path.abspath(os.path.join(test_dir,filename))
    dicdata = dic.import_2d(data=datafile, binary=False, layout='column', delimiter=",")

    raw_data = np.loadtxt(datafile, delimiter=",", skiprows=1)
    assert np.allclose(dicdata.ss_x, raw_data[:, 0]), "Mismatch in ss_x data column"
    assert np.allclose(dicdata.ss_y, raw_data[:, 1]), "Mismatch in ss_y data column"
    assert np.allclose(dicdata.u, raw_data[:, 2]), "Mismatch in u data column"
    assert np.allclose(dicdata.v, raw_data[:, 3]), "Mismatch in v data column"
    assert np.allclose(dicdata.mag, raw_data[:, 4]), "Mismatch in mag data column"
    assert np.allclose(dicdata.converged, raw_data[:, 5]), "Mismatch in cost data column"
    assert np.allclose(dicdata.cost, raw_data[:, 6]), "Mismatch in cost data column"
    assert np.allclose(dicdata.ftol, raw_data[:, 7]), "Mismatch in ftol data column"
    assert np.allclose(dicdata.xtol, raw_data[:, 8]), "Mismatch in xtol data column"
    assert np.allclose(dicdata.niter, raw_data[:, 9]), "Mismatch in niter data column"


def test_dic_data_matrix_import():

    filename = "./reference/ref_00_50.csv"
    datafile = os.path.abspath(os.path.join(test_dir,filename))
    dicdata = dic.import_2d(data=datafile, binary=False, layout='matrix', delimiter=",")

    raw_data = np.loadtxt(datafile, delimiter=",", skiprows=1)

    # loop over all x and y values in raw_data and check if they exist in the
    # matrix format of the dicdata
    for i in range(raw_data.shape[0]):
        x = raw_data[i, 0]
        y = raw_data[i, 1]
        u_val = raw_data[i, 2]
        try:
            row_idx, col_idx = np.where((dicdata.ss_x == x) & (dicdata.ss_y == y))
        except IndexError:
            raise ValueError(f"x={x}, y={y} not found in grid")

        u_matrix_val = dicdata.u[0, row_idx, col_idx]
        assert np.isclose(u_matrix_val, u_val), f"Mismatch at (x={x}, y={y}): {u_matrix_val} != {u_val}"
