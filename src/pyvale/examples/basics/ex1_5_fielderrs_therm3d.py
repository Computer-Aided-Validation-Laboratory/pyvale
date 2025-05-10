# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: Field-based systematic errors
--------------------------------------------------------------------------------
In this example we give an overview of field-based systematic errors. Field
errors require additional interpolation of the underlying physical field such as
uncertainty in a sensors position or sampling time. For this example we will
focus on field error sources that perturb sensor locations and sampling times.
In later examples we will analyse sensor orientation for vector and tensor
fields.

Note that field errors are more computationally intensive than basic errors as
they require additional interpolations of the underlying physical field.

Test case: Scalar field point sensors (thermocouples) on a 3D thermal simulation
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv


def main() -> None:
    # First we use everything we learned from the first three examples to build
    # a thermocouple sensor array for the same 3D thermal simulation we have
    # analysed in the previous examples. Then we will look at a new type of
    # systematic error called a field error which requires additional
    # interpolation of the underlying physical field to be measured.
    data_path = pyv.DataSet.thermal_3d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=None)

    n_sens = (1,4,1)
    x_lims = (12.5,12.5)
    y_lims = (0.0,33.0)
    z_lims = (0.0,12.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sample_times = np.linspace(0.0,np.max(sim_data.time),50) # | None

    sensor_data = pyv.SensorData(positions=sens_pos,
                                 sample_times=sample_times)

    field_key: str = "temperature"
    tc_array = pyv.SensorArrayFactory \
        .thermocouples_no_errs(sim_data,
                               sensor_data,
                               elem_dims=3,
                               field_name=field_key)

    # Now we will create a field error data class which we will use to build our
    # field error. This controls which sensor parameters will be perturbed such
    # as: position, time and orientation. Here we will perturb the sensor
    # positions on the face of the block using a normal distribution and we will
    # also perturb the measurement times.

    # We can apply a constant offset to each sensors position in x,y,z by
    # providing a shape=(num_sensors,coord[x,y,z]) array. Here we apply a
    # constant offset in the y and z direction for all sensors.
    pos_offset_xyz = np.array((0.0,1.0,1.0),dtype=np.float64)
    pos_offset_xyz = np.tile(pos_offset_xyz,(sens_pos.shape[0],1))

    # We can also apply a constant offset to the sampling times for all sensors
    time_offset = np.full((sample_times.shape[0],),0.1)

    # Using the `Gen*` random generators in pyvale we can randomly perturb the
    # position or sampling times of our virtual sensors.
    pos_rand = pyv.GenNormal(std=1.0) # units = mm
    time_rand = pyv.GenNormal(std=0.1) # units = s

    # Now we put everything into our field error data class ready to build our
    # field error object. Have a look at the other parameters in this data class
    # to geta feel for the other types of supported field errors. We will look
    # at the orientation and area averaging errors when we look at vector and
    # tensor fields in later examples.
    field_err_data = pyv.ErrFieldData(
        pos_offset_xyz=pos_offset_xyz,
        time_offset=time_offset,
        pos_rand_xyz=(None,pos_rand,pos_rand),
        time_rand=time_rand
    )

    # Adding our field error to our error chain is exactly the same as the basic
    # errors we have seen previously. We can also combine field errors with
    # basic errors and place them anywhere in our error chain. We can even chain
    # field errors together which we will look at in the next example. For now
    # we will just have a single field error so we can easily visualise what
    # this type of error does.
    err_chain = []

    # A field error needs to know which field it should interpolate for error
    # calculations so we provide the field from the sensor array as well as the
    # field error error data class.
    err_chain.append(pyv.ErrSysField(tc_array.get_field(),
                                     field_err_data))
    err_int = pyv.ErrIntegrator(err_chain,
                                sensor_data,
                                tc_array.get_measurement_shape())
    tc_array.set_error_integrator(err_int)


    # Now we can run the sensor simulation and display the results to see what
    # our field error has done.
    measurements = tc_array.calc_measurements()

    print(80*"-")

    sens_print: int = 0
    time_print: int = 5
    comp_print: int = 0

    print(f"These are the last {time_print} virtual measurements of sensor "
          + f"{sens_print}:")

    pyv.print_measurements(sens_array=tc_array,
                           sensors=(sens_print,sens_print+1),
                           components=(comp_print,comp_print+1),
                           time_steps=(measurements.shape[2]-time_print,
                                       measurements.shape[2]))
    print(80*"-")

    # We are going to save some figures to disk as well as displaying them
    # interactively so we create a directory for this:
    output_path = Path.cwd() / "pyvale-output"
    if not output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)


    # If we analyse the time traces we can see offsets in the sensor value and
    # the sampling times which we expect from our field error setup.
    (fig,ax) = pyv.plot_time_traces(tc_array,field_key)

    save_traces = output_path/"field_ex1_5_sensortraces.png"
    fig.savefig(save_traces, dpi=300, bbox_inches="tight")
    fig.savefig(save_traces.with_suffix(".svg"), dpi=300, bbox_inches="tight")

    plt.show()

    # It is also possible to view the perturbed sensor locations on the
    # simulation mesh if we create a plot after running the sensor simulation.
    pv_plot = pyv.plot_point_sensors_on_sim(tc_array,field_key)
    pv_plot.camera_position = [(59.354, 43.428, 69.946),
                               (-2.858, 13.189, 4.523),
                               (-0.215, 0.948, -0.233)]

    save_render = output_path / "fielderrs_ex1_5_sensorlocs.svg"
    pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex
    pv_plot.screenshot(save_render.with_suffix(".png"))

    pv_plot.show()

    # We have saved an image of the sensor traces and the perturbed locations of
    # the sensors to the `pyvale-output` directory in your current working
    # directory. Analyse these figures side by side to show that the location of
    # the perturbed sensor locations matches the expected sensor traces.


if __name__ == '__main__':
    main()
