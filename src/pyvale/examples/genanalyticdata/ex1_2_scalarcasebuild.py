# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import sympy
import pyvale

def main() -> None:

    case_data = pyvale.AnalyticData2D()
    case_data.length_x = 10.0
    case_data.length_y = 7.5
    n_elem_mult = 10
    case_data.num_elem_x = 4*n_elem_mult
    case_data.num_elem_y = 3*n_elem_mult
    case_data.time_steps = np.linspace(0.0,1.0,11)

    (sym_y,sym_x,sym_t) = sympy.symbols("y,x,t")
    case_data.funcs_x = (25.0 * sympy.sin(2*sympy.pi*sym_x/case_data.length_x),)
    case_data.funcs_y = (10.0/case_data.length_y * sym_y,)
    case_data.funcs_t = (sym_t,)
    case_data.offsets_space = (20.0,)
    case_data.offsets_time = (0.0,)


    data_gen = pyvale.AnalyticSimDataGen(case_data)
    sim_data = data_gen.generate_sim_data()

    (grid_x,grid_y,grid_field) = data_gen.get_visualisation_grid()

    fig, ax = plt.subplots()
    cs = ax.contourf(grid_x,grid_y,grid_field)
    cbar = fig.colorbar(cs)
    plt.axis('scaled')
    plt.show()


if __name__ == '__main__':
    main()
