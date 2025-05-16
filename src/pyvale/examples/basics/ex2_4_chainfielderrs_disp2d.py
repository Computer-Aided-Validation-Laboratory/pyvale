# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: Chaining field errors
----------------------------------------------------------------------------
In this example we show how field errors can be chained together and accumulated
allowing for successive perturbations in postion, sampling time and orientation.
It is more computationally efficient to provide a single field error object as
this will perform all perturbations in a single step allowing for a single new
interpolation of the underlying physical field. However, in some cases it can
be useful to separate the sensor parameter perturbations to determine which is
contributing most to the total error.

Note that this tutorial assumes you are familiar with the use of pyvale for
scalar fields as described in the first set of examples.

Test case: point displacement sensors on a 2D plate with hole loaded in tension
"""

import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv

def main() -> None:
    # We start by building the same displacement sensor array applied to a 2D
    # solid mechanics simulation that we have analysed previously.
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
    sensor_positions = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sensor_data = pyv.SensorData(positions=sensor_positions,
                                 sample_times=sample_times)

    disp_sens_array = pyv.SensorArrayPoint(sensor_data,
                                           disp_field,
                                           descriptor)


    # Now we will build a series of field errors that cause succesive offsets in
    # sensor sampling time, sensor position and sensor orientation. That way
    # we should be able to analyse the sensor data object at each point in the
    # error chain to see how the sensor parameters have accumulated.

    # We will apply a position offset of -1.0mm in the x and y axes.
    pos_offset = -1.0*np.ones_like(sensor_positions)
    pos_offset[:,2] = 0.0 # in 2d we only have offset in x and y so zero z
    pos_error_data = pyv.ErrFieldData(pos_offset_xyz=pos_offset)

    # We will apply a rotation offset about the z axis of 1 degree
    angle_offset = np.zeros_like(sensor_positions)
    angle_offset[:,0] = 1.0 # only rotate about z in 2D
    angle_error_data = pyv.ErrFieldData(ang_offset_zyx=angle_offset)

    time_offset = 2.0*np.ones_like(disp_sens_array.get_sample_times())
    time_error_data = pyv.ErrFieldData(time_offset=time_offset)

    # Now we add all our field errors to our error chain. We add each error
    # twice to see how they accumulate with each other. We also need to set the
    # error dependence to `DEPENDENT` so that the sensor state is accumulated
    # over the error chain as field errors are `INDEPENDENT` by default.
    err_chain = []
    err_chain.append(pyv.ErrSysField(disp_field,
                                     time_error_data,
                                     pyv.EErrDep.DEPENDENT))
    err_chain.append(pyv.ErrSysField(disp_field,
                                     time_error_data,
                                     pyv.EErrDep.DEPENDENT))

    err_chain.append(pyv.ErrSysField(disp_field,
                                     pos_error_data,
                                     pyv.EErrDep.DEPENDENT))
    err_chain.append(pyv.ErrSysField(disp_field,
                                     pos_error_data,
                                     pyv.EErrDep.DEPENDENT))

    err_chain.append(pyv.ErrSysField(disp_field,
                                     angle_error_data,
                                     pyv.EErrDep.DEPENDENT))
    err_chain.append(pyv.ErrSysField(disp_field,
                                     angle_error_data,
                                     pyv.EErrDep.DEPENDENT))

    # Instead of setting the dependence for each individual error above we could
    # also just use our error integration options to force all errors to be
    # `DEPENDENT`. We also set the error integration options to store the errors
    # for each step in the error chain so we can analyse the sensor data at each
    # step of chain. This option also allows us to separate the contribution of
    # each error in the chain to the total error rather than just being able to
    # analyse the total systematic and total random error which is the default.
    # Note that this option will use more memory.
    err_int_opts = pyv.ErrIntOpts(force_dependence=pyv.EErrDep.DEPENDENT,
                                  store_all_errs=True)

    # Now we build our error integrator, add it to our sensor array and then run
    # our sensor simulation to obtain some virtual measurements.
    error_int = pyv.ErrIntegrator(err_chain,
                                  sensor_data,
                                  disp_sens_array.get_measurement_shape(),
                                  err_int_opts)
    disp_sens_array.set_error_integrator(error_int)

    measurements = disp_sens_array.calc_measurements()


    # Here we will print to the console the time, position and angle of from the
    # sensor data objects at each point in the error chain. We should see each
    # sensor parameter perturbed and accumulated throughout the chain:
    sens_data_by_chain = error_int.get_sens_data_by_chain()
    if sens_data_by_chain is not None:
        for ii,ss in enumerate(sens_data_by_chain):
            print(80*"-")
            if ss is not None:
                print(f"SensorData @ [{ii}]")
                print("TIME")
                print(ss.sample_times)
                print()
                print("POSITIONS")
                print(ss.positions)
                print()
                print("ANGLES")
                for aa in ss.angles:
                    print(aa.as_euler("zyx",degrees=True))
                print()
            print(80*"-")

    # Try setting all the field errors to be `INDEPENDENT` using the error
    # integration options above. You should see that the sensor parameters are
    # not accumulated throughout the error chain.

    # Here we print the final sampling time, sensor positions and sensor angles
    # at the end of error chain.
    print()
    print(80*"=")
    sens_data_accumulated = error_int.get_sens_data_accumulated()
    print("TIME")
    print(sens_data_accumulated.sample_times)
    print()
    print("POSITIONS")
    print(sens_data_accumulated.positions)
    print()
    print("ANGLES")
    for aa in sens_data_accumulated.angles:
        print(aa.as_euler("zyx",degrees=True))
    print()
    print(80*"=")

    # We print the results for one of the sensors so we can see what the errors
    # are for the last few sampling times.
    print(80*"-")

    sens_print: int = 0
    time_print: int = 5
    comp_print: int = 0

    print("ROTATED SENSORS WITH ANGLE ERRORS:")
    print(f"These are the last {time_print} virtual measurements of sensor "
          + f"{sens_print} for {field_comps[comp_print]}:")

    pyv.print_measurements(sens_array=disp_sens_array,
                           sensors=(sens_print,sens_print+1),
                           components=(comp_print,comp_print+1),
                           time_steps=(measurements.shape[2]-time_print,
                                       measurements.shape[2]))
    print(80*"-")


    # Finally, we plot the time traces for all field components.
    for ff in field_comps:
        pyv.plot_time_traces(disp_sens_array,ff)

    plt.show()


if __name__ == "__main__":
    main()