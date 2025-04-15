"""
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv


def main() -> None:
    """pyvale example: thermocouples on a 2d plate
    ----------------------------------------------------------------------------
    - Demonstrates area averaging for truth and for systematic errors
    """
    data_path = pyv.DataSet.thermal_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale m to mm to make 3D visualisation scaling correct for pyvista
    sim_data = pyv.scale_length_units(1000.0,sim_data)

    descriptor = pyv.SensorDescriptorFactory.temperature_descriptor()

    field_key = "temperature"
    t_field = pyv.FieldScalar(sim_data,
                                 field_key=field_key,
                                 spat_dims=2)

    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = True
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sensor_dims = np.array([10.0,10.0,0])
    sensor_data = pyv.SensorData(positions=sens_pos,
                                    sample_times=sample_times,
                                    spatial_averager=pyv.EIntSpatialType.QUAD4PT,
                                    spatial_dims=sensor_dims)

    tc_array = pyv.SensorArrayPoint(sensor_data,
                                       t_field,
                                       descriptor)

    area_avg_err_data = pyv.ErrFieldData(
        spatial_averager=pyv.EIntSpatialType.RECT1PT,
        spatial_dims=sensor_dims
    )
    err_chain = []
    err_chain.append(pyv.ErrSysField(t_field,
                                        area_avg_err_data))
    error_int = pyv.ErrIntegrator(err_chain,
                                     sensor_data,
                                     tc_array.get_measurement_shape())
    tc_array.set_error_integrator(error_int)

    measurements = tc_array.get_measurements()

    print("\n"+80*"-")
    print("For a sensor: measurement = truth + sysematic error + random error")
    print(f"measurements.shape = {measurements.shape} = "+
          "(n_sensors,n_field_components,n_timesteps)\n")
    print("The truth, systematic error and random error arrays have the same "+
          "shape.")

    print(80*"-")
    print("Looking at the last 5 time steps (measurements) of sensor 0:")
    pyv.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (0,10))
    print(80*"-")

    pyv.plot_time_traces(tc_array,field_key)
    plt.show()


if __name__ == "__main__":
    main()
