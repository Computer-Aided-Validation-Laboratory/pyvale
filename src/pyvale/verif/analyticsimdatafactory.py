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
from pyvale.verif.analyticsimdatagenerator import (AnalyticData2D,
                                                   AnalyticSimDataGen)


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


def scalar_linear_2d() -> tuple[mh.SimData,AnalyticSimDataGen]:
    """_summary_

    Returns
    -------
    tuple[mh.SimData,AnalyticSimDataGenerator]
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


def scalar_quadratic_2d() -> tuple[mh.SimData,AnalyticSimDataGen]:
    """_summary_

    Returns
    -------
    tuple[mh.SimData,AnalyticSimDataGenerator]
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


def vector_linear_2d() -> tuple[mh.SimData,AnalyticSimDataGen]:
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


def tensor_linear_2d() -> tuple[mh.SimData,AnalyticSimDataGen]:
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
    return (sim_data,data_gen)

