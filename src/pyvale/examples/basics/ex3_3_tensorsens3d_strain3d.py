# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: Custom tensor field sensors (strain gauges) in 3D
--------------------------------------------------------------------------------
In this example we build a custom tensor field sensor array (i.e. a strain gauge
array) in 3D. We will also demonstrate how to specify sensor angles and field
errors based on sensor angles.

Note that this tutorial assumes you are familiar with the use of pyvale for
scalar fields as described in the first set of examples.

Test case: point strain sensors on a 2D plate with hole loaded in tension
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import mooseherder as mh
import pyvale as pyv

def main() -> None:

    # Frist we load our simulation as a `SimData` object. In this case we are
    # loading a 10mm cube loaded in tension in the y direction with the addition
    # of a thermal gradient in the y direction.
    data_path = pyv.DataSet.element_case_path(pyv.EElemTest.HEX20)
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    # As we are creating a 3D tensor field sensor we now have a third
    # displacement field component here for scaling. Note that you don't need to
    # scale the displacements here if you only want to analyse strains.
    disp_comps = ("disp_x","disp_y","disp_z")
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=disp_comps)

    # Here is the main difference when creating a tensor field sensor array. We
    # create a tensor field where we need to specify the normal and deviatoric
    # component string keys as they appear in our `SimData` object. We have a 3D
    # simulation here so we have 3 normal components and 3 deviatoric (shear).
    field_name = "strain"
    norm_comps = ("strain_xx","strain_yy","strain_zz")
    dev_comps = ("strain_xy","strain_yz","strain_xz")
    strain_field = pyv.FieldTensor(sim_data,
                                   field_name=field_name,
                                   norm_comps=norm_comps,
                                   dev_comps=dev_comps,
                                   elem_dims=3)

    # Here we manually define our sensor positions to place a sensor on the
    # centre of each face of our 10mm cube. From here everything is the same as
    # for our 2D vector field sensor arrays.
    sensor_positions = np.array(((5.0,0.0,5.0),     # bottom
                                 (5.0,10.0,5.0),    # top
                                 (5.0,5.0,0.0),     # xy face
                                 (5.0,5.0,10.0),    # xy face
                                 (0.0,5.0,5.0),     # yz face
                                 (10.0,5.0,5.0),))  # yz face

    # We set custom sensor sampling times here but we could also set this to
    # None to have the sensors sample at the simulation time steps.
    sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    # We are going to manually specify the sensor angles for all our sensors.
    sens_angles = (Rotation.from_euler("zyx", [0, 0, 0], degrees=True),
                   Rotation.from_euler("zyx", [0, 0, 0], degrees=True),
                   Rotation.from_euler("zyx", [45, 0, 0], degrees=True),
                   Rotation.from_euler("zyx", [45, 0, 0], degrees=True),
                   Rotation.from_euler("zyx", [0, 0, 45], degrees=True),
                   Rotation.from_euler("zyx", [0, 0, 45], degrees=True),)


    sens_data = pyv.SensorData(positions=sensor_positions,
                               sample_times=sample_times,
                               angles=sens_angles)

    # Here we create a descriptor that will be used to label visualisations of
    # the sensor locations and time traces for our sensors. For the strain
    # gauges we are modelling here we could also use the descriptor factory to
    # get these defaults.
    descriptor = pyv.SensorDescriptor(name="Strain",
                                      symbol=r"\varepsilon",
                                      units=r"-",
                                      tag="SG",
                                      components=('xx','yy','zz','xy','yz','xz'))


    straingauge_array = pyv.SensorArrayPoint(sens_data,
                                             strain_field,
                                             descriptor)

    # We can add any errors we like to our error chain. Here we add some basic
    # percentage errors.
    error_chain = []
    error_chain.append(pyv.ErrSysUnif(low=-0.1e-3,high=0.1e-3))
    error_chain.append(pyv.ErrRandNormPercent(std_percent=1.0))

    # Now we add a field error to perturb the positions of each sensor on its
    # relevant face and then add a +/- 2deg angle error.

    pos_uncert = 0.1 # units = mm
    pos_rand_xyz = (pyv.GenNormal(std=pos_uncert),
                    pyv.GenNormal(std=pos_uncert),
                    pyv.GenNormal(std=pos_uncert))

    angle_uncert = 2.0
    angle_rand_zyx = (pyv.GenUniform(low=-angle_uncert,high=angle_uncert), # units = deg
                      pyv.GenUniform(low=-angle_uncert,high=angle_uncert),
                      pyv.GenUniform(low=-angle_uncert,high=angle_uncert))

    # We are going to lock position perturbation so that the sensors stay on the
    # faces of the cube they are positioned on.
    pos_lock = np.full(sensor_positions.shape,False,dtype=bool)
    pos_lock[0:2,1] = True   # Block translation in y
    pos_lock[2:4,2] = True   # Block translation in z
    pos_lock[4:6,0] = True   # Block translation in x

    # We are also going to lock angular perturbation so that each sensor is only
    # allowed to rotate on the plane it is on.
    angle_lock = np.full(sensor_positions.shape,True,dtype=bool)
    angle_lock[0:2,1] = False   # Allow rotation about y
    angle_lock[2:4,0] = False   # Allow rotation about z
    angle_lock[4:6,2] = False   # Alloq rotation about x

    field_error_data = pyv.ErrFieldData(pos_rand_xyz=pos_rand_xyz,
                                        pos_lock_xyz=pos_lock,
                                        ang_rand_zyx=angle_rand_zyx,
                                        ang_lock_zyx=angle_lock)
    sys_err_field = pyv.ErrSysField(strain_field,field_error_data)
    error_chain.append(sys_err_field)


    error_int = pyv.ErrIntegrator(error_chain,
                                  sens_data,
                                  straingauge_array.get_measurement_shape())
    straingauge_array.set_error_integrator(error_int)

    # We run our virtual sensor simulation as normal. The only thing to note is
    # that the second dimension of our measurement array will contain our tensor
    # components in the order they are specified in the tuples with the normal
    # components first followed by the deviatoric.
    measurements = straingauge_array.calc_measurements()

    # We print some of the results for one of the sensors so we can see the
    # effect of the field errors.
    print(80*"-")

    sens_print: int = 0
    time_print: int = 5
    comp_print: int = 1 # strain_yy based on order in tuple

    print("ROTATED SENSORS WITH ANGLE ERRORS:")
    print(f"These are the last {time_print} virtual measurements of sensor "
          + f"{sens_print} for {norm_comps[comp_print]}:")

    pyv.print_measurements(sens_array=straingauge_array,
                           sensors=(sens_print,sens_print+1),
                           components=(comp_print,comp_print+1),
                           time_steps=(measurements.shape[2]-time_print,
                                       measurements.shape[2]))
    print(80*"-")

    # We can plot a given component of our tensor field and display our sensor
    # locations with respect to the field.
    plot_field = "strain_yy"
    pv_plot = pyv.plot_point_sensors_on_sim(straingauge_array,plot_field)
    pv_plot.show(cpos="xy")

    # We can also plot time traces for all components of the tensor field.
    for cc in (norm_comps+dev_comps):
        pyv.plot_time_traces(straingauge_array,cc)

    plt.show()


if __name__ == "__main__":
    main()