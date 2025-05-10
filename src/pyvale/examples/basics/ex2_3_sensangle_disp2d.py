# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: Sensor angles for vector fields
--------------------------------------------------------------------------------
In this example we demonstrate how to setup vector field sensors at custom
orientations with respect to the simulation coordinate system. We first build a
sensor array aligned with the simulation coords in the same way as the previous
example. We then build a sensor array with the sensors rotated and compare this
to the case with no rotation.

Note that this tutorial assumes you are familiar with the use of pyvale for
scalar fields as described in the first set of examples.

Test case: point displacement sensors on a 2D plate with hole loaded in tension
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import mooseherder as mh
import pyvale as pyv

def main() -> None:
    # First we are going to setup the same displacement sensor array on the 2D
    # solid mechanics test case we have used previously. This will serve as a
    # baseline with no sensor rotation.

    data_path = pyv.DataSet.mechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    field_name = "disp"
    field_comps = ("disp_x","disp_y")
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=field_comps)

    descriptor = pyv.SensorDescriptorFactory.displacement_descriptor()

    disp_field = pyv.FieldVector(sim_data,field_name,field_comps,elem_dims=2)

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)


    sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data_norot = pyv.SensorData(positions=sens_pos,
                                     sample_times=sample_times)

    disp_sens_norot = pyv.SensorArrayPoint(sens_data_norot,
                                           disp_field,
                                           descriptor)

    disp_sens_norot.calc_measurements()

    # To create our sensor array with rotated sensors we need to add a tuple of
    # scipy rotation objects to our sensor data class. This tuple must be the
    # same length as the number of sensors in the sensor array. Note that it is
    # also possible to specify a single rotation in the tuple in this case all
    # sensors are assumed to have the same rotation and they are batch processed
    # to increase speed. Here we will define our rotations to all be the same
    # rotation in degrees about the z axis which is the out of plane axis for
    # our current test case.
    sens_angles = sens_pos.shape[0] * \
        (Rotation.from_euler("zyx", [45, 0, 0], degrees=True),)

    # We could have also use a single element tuple to have all sensors have the
    # angle and batch process them:
    sens_angles = (Rotation.from_euler("zyx", [45, 0, 0], degrees=True),)


    sens_data_rot = pyv.SensorData(positions=sens_pos,
                                   sample_times=sample_times,
                                   angles=sens_angles)

    disp_sens_rot = pyv.SensorArrayPoint(sens_data_rot,
                                            disp_field,
                                            descriptor)

    # We can also use a field error to add uncertainty to the sensors angle.
    # We can apply a specific offset to each sensor or provide a random
    # generator to perturb the sensors orientation. Note that the offset and
    # the random generator should provide the perturbation in degrees.
    angle_offset = np.zeros_like(sens_pos)
    angle_offset[:,0] = 2.0 # only rotate about z in 2D
    angle_rand = (pyv.GenNormal(std=2.0),None,None)
    angle_error_data = pyv.ErrFieldData(ang_offset_zyx=angle_offset,
                                        ang_rand_zyx=angle_rand)


    sys_err_rot = pyv.ErrSysField(disp_field,angle_error_data)
    sys_err_int = pyv.ErrIntegrator([sys_err_rot],
                                    sens_data_rot,
                                    disp_sens_rot.get_measurement_shape())
    disp_sens_rot.set_error_integrator(sys_err_int)

    measurements = disp_sens_rot.calc_measurements()


    # We print some of the results for one of the sensors so we can see the
    # effect of the field errors.
    print(80*"-")

    sens_print: int = 0
    time_print: int = 5
    comp_print: int = 0

    print("ROTATED SENSORS WITH ANGLE ERRORS:")
    print(f"These are the last {time_print} virtual measurements of sensor "
          + f"{sens_print} for {field_comps[comp_print]}:")

    pyv.print_measurements(sens_array=disp_sens_rot,
                           sensors=(sens_print,sens_print+1),
                           components=(comp_print,comp_print+1),
                           time_steps=(measurements.shape[2]-time_print,
                                       measurements.shape[2]))
    print(80*"-")

    # We can now plot the traces for the non-rotated and rotated sensors to
    # compare them:
    for ff in field_comps:
        (_,ax) = pyv.plot_time_traces(disp_sens_norot,ff)
        ax.set_title("No Rotation")
        (_,ax) = pyv.plot_time_traces(disp_sens_rot,ff)
        ax.set_title("Rotated with Errors")

    plt.show()


if __name__ == "__main__":
    main()