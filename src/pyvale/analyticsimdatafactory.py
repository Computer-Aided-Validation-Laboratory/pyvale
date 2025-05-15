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
import mooseherder as mh
from pyvale.analyticsimdatagenerator import (AnalyticData2D,
                                                  AnalyticSimDataGen)


def standard_case_2d() -> AnalyticData2D:
    """Created the standard 2D analytic test case which is a plate with
    dimensions 10x7.5 (x,y), number of elements 40x30 (x,y), and time steps of
    0 to 10 in increments of 1.

    Returns
    -------
    AnalyticCaseData2D
        _description_
    """
    case_data = AnalyticData2D()
    case_data.length_x = 10.0
    case_data.length_y = 7.5
    n_elem_mult = 10
    case_data.num_elem_x = 4*n_elem_mult
    case_data.num_elem_y = 3*n_elem_mult
    case_data.time_steps = np.linspace(0.0,1.0,11)
    return case_data


class AnalyticCaseFactory:
    """Namespace for function used to build pre-defined 2D meshes and fields
    based on analytic functions for testing the sensor simulation functionality
    of pyvale.
    """

    @staticmethod
    def scalar_linear_2d() -> tuple[mh.SimData,AnalyticSimDataGen]:
        """_summary_

        Returns
        -------
        tuple[mh.SimData,AnalyticSimDataGenerator]
            _description_
        """
        case_data = standard_case_2d()
        (sym_y,sym_x,sym_t) = sympy.symbols("y,x,t")
        case_data.funcs_x = (20.0/case_data.length_x * sym_x,)
        case_data.funcs_y = (10.0/case_data.length_y * sym_y,)
        case_data.funcs_t = (sym_t,)
        case_data.offsets_space = (20.0,)
        case_data.offsets_time = (0.0,)

        data_gen = AnalyticSimDataGen(case_data)

        sim_data = data_gen.generate_sim_data()

        return (sim_data,data_gen)

    @staticmethod
    def scalar_quadratic_2d() -> tuple[mh.SimData,AnalyticSimDataGen]:
        """_summary_

        Returns
        -------
        tuple[mh.SimData,AnalyticSimDataGenerator]
            _description_
        """
        case_data = standard_case_2d()
        (sym_y,sym_x,sym_t) = sympy.symbols("y,x,t")
        case_data.funcs_x = (sym_x*(sym_x - case_data.length_x),)
        case_data.funcs_y = (sym_y*(sym_y - case_data.length_y),)
        case_data.funcs_t = (sym_t,)

        data_gen = AnalyticSimDataGen(case_data)

        sim_data = data_gen.generate_sim_data()

        return (sim_data,data_gen)




