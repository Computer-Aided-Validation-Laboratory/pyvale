# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import matplotlib.pyplot as plt
import pyvale

def main() -> None:

    (sim_data,data_gen) = pyvale.AnalyticCaseFactory.scalar_linear_2d()

    (grid_x,grid_y,grid_field) = data_gen.get_visualisation_grid()

    fig, ax = plt.subplots()
    cs = ax.contourf(grid_x,grid_y,grid_field)
    cbar = fig.colorbar(cs)
    plt.axis('scaled')


    (sim_data,data_gen) = pyvale.AnalyticCaseFactory.scalar_quadratic_2d()

    (grid_x,grid_y,grid_field) = data_gen.get_visualisation_grid()

    fig, ax = plt.subplots()
    cs = ax.contourf(grid_x,grid_y,grid_field)
    cbar = fig.colorbar(cs)
    plt.axis('scaled')

    plt.show()


if __name__ == '__main__':
    main()
