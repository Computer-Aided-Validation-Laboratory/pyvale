'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np

from pyvale.sensorarraypoint import SensorArrayPoint


def create_sensor_pos_array(n_sens: tuple[int,int,int],
                           x_lims: tuple[float, float],
                           y_lims: tuple[float, float],
                           z_lims: tuple[float, float]) -> np.ndarray:

    sens_pos_x = np.linspace(x_lims[0],x_lims[1],n_sens[0]+2)[1:-1]
    sens_pos_y = np.linspace(y_lims[0],y_lims[1],n_sens[1]+2)[1:-1]
    sens_pos_z = np.linspace(z_lims[0],z_lims[1],n_sens[2]+2)[1:-1]

    (sens_grid_x,sens_grid_y,sens_grid_z) = np.meshgrid(
        sens_pos_x,sens_pos_y,sens_pos_z)

    sens_pos_x = sens_grid_x.flatten()
    sens_pos_y = sens_grid_y.flatten()
    sens_pos_z = sens_grid_z.flatten()

    sens_pos = np.vstack((sens_pos_x,sens_pos_y,sens_pos_z)).T
    return sens_pos


def print_measurements(sens_array: SensorArrayPoint,
                       sensors: tuple[int,int],
                       components: tuple[int,int],
                       time_steps: tuple[int,int])  -> None:

    measurement =  sens_array.get_measurements()
    truth = sens_array.get_truth()
    rand_errs = sens_array.get_random_errors()
    sys_errs = sens_array.get_systematic_errors()

    print(f"\nmeasurement.shape = \n    {measurement.shape}")
    print_meas = measurement[sensors[0]:sensors[1],
                             components[0]:components[1],
                             time_steps[0]:time_steps[1]]
    print(f"measurement = \n    {print_meas}")

    print_truth = truth[sensors[0]:sensors[1],
                        components[0]:components[1],
                        time_steps[0]:time_steps[1]]
    print(f"truth = \n    {print_truth}")


    print_randerrs = rand_errs[sensors[0]:sensors[1],
                                components[0]:components[1],
                                time_steps[0]:time_steps[1]]
    print(f"rand_errs = \n    {print_randerrs}")


    print_syserrs = sys_errs[sensors[0]:sensors[1],
                                    components[0]:components[1],
                                    time_steps[0]:time_steps[1]]
    print(f"sys_errs = \n    {print_syserrs}")

    print()
