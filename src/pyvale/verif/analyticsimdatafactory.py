# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Helper functions and mini factory for building standard test meshes with
analytic functions for the physical fields.
"""

import numpy as np
import sympy
import pyvale.mooseherder as mh
import pyvale.dataio as io
from pyvale.verif.analyticsimdatagenerator import (
    AnalyticData2D,
    AnalyticData3D,
    AnalyticSimDataGen,
)


def standard_case_2d(field_keys: tuple[str,...]) -> AnalyticData2D:
    """Created the standard 2D analytic test case which is a plate with
    dimensions 10x7.5 (x,y), number of elements 40x30 (x,y), and time steps of
    0 to 10 in increments of 1.

    Returns
    -------
    AnalyticCaseData2D
        _description_
    """
    case_data = AnalyticData2D(field_keys = field_keys)
    case_data.length_x = 10.0
    case_data.length_y = 7.5
    n_elem_mult = 10
    case_data.num_elem_x = 4*n_elem_mult
    case_data.num_elem_y = 3*n_elem_mult
    case_data.time_steps = np.linspace(0.0,1.0,11)
    return case_data


def scalar_linear_2d() -> tuple[io.SimData,AnalyticSimDataGen]:
    """_summary_

    Returns
    -------
    tuple[io.SimData,AnalyticSimDataGenerator]
        _description_
    """
    field_key = "temperature"
    case_data = standard_case_2d((field_key,))

    (sym_y,sym_x,sym_t) = sympy.symbols("y,x,t")

    case_data.funcs_x = {field_key: 20.0/case_data.length_x * sym_x,}
    case_data.funcs_y = {field_key: 10.0/case_data.length_y * sym_y,}
    case_data.funcs_t = {field_key: sym_t,}
    case_data.offset_space_x = {field_key: 20.0,}
    case_data.offset_time = {field_key: 0.0,}

    data_gen = AnalyticSimDataGen(case_data)

    sim_data = data_gen.generate_sim_data()

    return (sim_data,data_gen)


def scalar_quadratic_2d() -> tuple[io.SimData,AnalyticSimDataGen]:
    """_summary_

    Returns
    -------
    tuple[io.SimData,AnalyticSimDataGenerator]
        _description_
    """
    field_key = "temperature"
    case_data = standard_case_2d((field_key,))

    (sym_y,sym_x,sym_t) = sympy.symbols("y,x,t")

    case_data.funcs_x = {field_key: sym_x*(sym_x - case_data.length_x),}
    case_data.funcs_y = {field_key: sym_y*(sym_y - case_data.length_y),}
    case_data.funcs_t = {field_key: sym_t,}

    data_gen = AnalyticSimDataGen(case_data)

    sim_data = data_gen.generate_sim_data()

    return (sim_data,data_gen)


def vector_linear_2d() -> tuple[io.SimData,AnalyticSimDataGen]:
    field_keys = ("disp_x","disp_y")
    case_data = standard_case_2d(field_keys)

    (sym_y,sym_x,sym_t) = sympy.symbols("y,x,t")

    for kk in field_keys:
        case_data.funcs_x[kk] = 20.0/case_data.length_x * sym_x
        case_data.funcs_y[kk] = 10.0/case_data.length_y * sym_y
        case_data.funcs_t[kk] = sym_t
        case_data.offset_space_x[kk] = 20.0
        case_data.offset_space_y[kk] = 0.0
        case_data.offset_time[kk] = 0.0

    data_gen = AnalyticSimDataGen(case_data)
    sim_data = data_gen.generate_sim_data()
    return (sim_data,data_gen)


def tensor_linear_2d() -> tuple[io.SimData,AnalyticSimDataGen]:
    field_keys = ("strain_xx","strain_yy","strain_xy")
    case_data = standard_case_2d(field_keys)

    (sym_y,sym_x,sym_t) = sympy.symbols("y,x,t")

    for kk in field_keys:
        case_data.funcs_x[kk] = 20.0/case_data.length_x * sym_x
        case_data.funcs_y[kk] = 10.0/case_data.length_y * sym_y
        case_data.funcs_t[kk] = sym_t
        case_data.offset_space_x[kk] = 20.0
        case_data.offset_space_y[kk] = 0.0
        case_data.offset_time[kk] = 0.0

    data_gen = AnalyticSimDataGen(case_data)
    sim_data = data_gen.generate_sim_data()
    return (sim_data, data_gen)


def standard_case_3d(field_keys: tuple[str, ...]) -> AnalyticData3D:
    """Creates standard 3D analytic test case: a box with dimensions
    10x7.5x5.0 (x,y,z), elements 40x30x20, and time steps 0 to 1 in 11 steps.
    """
    case_data = AnalyticData3D(field_keys=field_keys)
    case_data.length_x = 10.0
    case_data.length_y = 7.5
    case_data.length_z = 5.0
    n_elem_mult = 5
    case_data.num_elem_x = 4 * n_elem_mult
    case_data.num_elem_y = 3 * n_elem_mult
    case_data.num_elem_z = 2 * n_elem_mult
    case_data.time_steps = np.linspace(0.0, 1.0, 11)
    return case_data


def scalar_linear_3d() -> tuple[io.SimData, AnalyticSimDataGen]:
    """Generates 3D SimData with linear scalar field in x, y, z and time."""
    field_key = "temperature"
    case_data = standard_case_3d((field_key,))

    sym_z, sym_y, sym_x, sym_t = sympy.symbols("z,y,x,t")

    case_data.funcs_x = {field_key: 20.0 / case_data.length_x * sym_x}
    case_data.funcs_y = {field_key: 10.0 / case_data.length_y * sym_y}
    case_data.funcs_z = {field_key: 5.0 / case_data.length_z * sym_z}
    case_data.funcs_t = {field_key: sym_t}
    case_data.offset_space_x = {field_key: 20.0}
    case_data.offset_space_y = {field_key: 0.0}
    case_data.offset_space_z = {field_key: 0.0}
    case_data.offset_time = {field_key: 0.0}

    data_gen = AnalyticSimDataGen(case_data)
    sim_data = data_gen.generate_sim_data()
    return (sim_data, data_gen)


def scalar_quadratic_3d() -> tuple[io.SimData, AnalyticSimDataGen]:
    """Generates 3D SimData with quadratic scalar field in x, y, z and time."""
    field_key = "temperature"
    case_data = standard_case_3d((field_key,))

    sym_z, sym_y, sym_x, sym_t = sympy.symbols("z,y,x,t")

    case_data.funcs_x = {field_key: sym_x * (sym_x - case_data.length_x)}
    case_data.funcs_y = {field_key: sym_y * (sym_y - case_data.length_y)}
    case_data.funcs_z = {field_key: sym_z * (sym_z - case_data.length_z)}
    case_data.funcs_t = {field_key: sym_t}

    data_gen = AnalyticSimDataGen(case_data)
    sim_data = data_gen.generate_sim_data()
    return (sim_data, data_gen)


def vector_linear_3d() -> tuple[io.SimData, AnalyticSimDataGen]:
    """Generates 3D SimData with linear vector field in x, y, z and time."""
    field_keys = ("disp_x", "disp_y", "disp_z")
    case_data = standard_case_3d(field_keys)

    sym_z, sym_y, sym_x, sym_t = sympy.symbols("z,y,x,t")

    for kk in field_keys:
        case_data.funcs_x[kk] = 20.0 / case_data.length_x * sym_x
        case_data.funcs_y[kk] = 10.0 / case_data.length_y * sym_y
        case_data.funcs_z[kk] = 5.0 / case_data.length_z * sym_z
        case_data.funcs_t[kk] = sym_t
        case_data.offset_space_x[kk] = 20.0
        case_data.offset_space_y[kk] = 0.0
        case_data.offset_space_z[kk] = 0.0
        case_data.offset_time[kk] = 0.0

    data_gen = AnalyticSimDataGen(case_data)
    sim_data = data_gen.generate_sim_data()
    return (sim_data, data_gen)


def tensor_linear_3d() -> tuple[io.SimData, AnalyticSimDataGen]:
    """Generates 3D SimData with linear tensor field in x, y, z and time."""
    field_keys = (
        "strain_xx",
        "strain_yy",
        "strain_zz",
        "strain_yz",
        "strain_xz",
        "strain_xy",
    )
    case_data = standard_case_3d(field_keys)

    sym_z, sym_y, sym_x, sym_t = sympy.symbols("z,y,x,t")

    for kk in field_keys:
        case_data.funcs_x[kk] = 20.0 / case_data.length_x * sym_x
        case_data.funcs_y[kk] = 10.0 / case_data.length_y * sym_y
        case_data.funcs_z[kk] = 5.0 / case_data.length_z * sym_z
        case_data.funcs_t[kk] = sym_t
        case_data.offset_space_x[kk] = 20.0
        case_data.offset_space_y[kk] = 0.0
        case_data.offset_space_z[kk] = 0.0
        case_data.offset_time[kk] = 0.0

    data_gen = AnalyticSimDataGen(case_data)
    sim_data = data_gen.generate_sim_data()
    return (sim_data, data_gen)


