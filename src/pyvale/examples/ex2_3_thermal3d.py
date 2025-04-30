# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv


def main() -> None:
    """pyvale example: thermocouples on a 3D divertor monoblock heatsink
    ----------------------------------------------------------------------------
    """
    data_path = pyv.DataSet.thermal_3d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    use_auto_descriptor = "manual"
    if use_auto_descriptor == "factory":
        descriptor = pyv.SensorDescriptorFactory.temperature_descriptor()
    elif use_auto_descriptor == "manual":
        descriptor = pyv.SensorDescriptor()
        descriptor.name = "Temperature"
        descriptor.symbol = "T"
        descriptor.units = r"^{\circ}C"
        descriptor.tag = "TC"
    else:
        descriptor = pyv.SensorDescriptor()


    field_key = "temperature"
    t_field = pyv.FieldScalar(sim_data,
                                 field_key=field_key,
                                 spat_dims=3)

    n_sens = (1,4,1)
    x_lims = (12.5,12.5)
    y_lims = (0.0,33.0)
    z_lims = (0.0,12.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),80)

    sens_data = pyv.SensorData(positions=sens_pos,
                                  sample_times=sample_times)

    tc_array = pyv.SensorArrayPoint(sens_data,
                                       t_field,
                                       descriptor)

    errors_on = {"indep_sys": True,
                 "rand": True,
                 "dep_sys": True}

    error_chain = []
    if errors_on["indep_sys"]:
        error_chain.append(pyv.ErrSysOffset(offset=-5.0))
        error_chain.append(pyv.ErrSysUniform(low=-10.0,
                                                   high=10.0))

    if errors_on["rand"]:
        error_chain.append(pyv.ErrRandNormPercent(std_percent=5.0))
        error_chain.append(pyv.ErrRandUnifPercent(low_percent=-5.0,
                                                   high_percent=5.0))

    if errors_on["dep_sys"]:
        error_chain.append(pyv.ErrSysDigitisation(bits_per_unit=1/20))
        error_chain.append(pyv.ErrSysSaturation(meas_min=0.0,meas_max=800.0))

    if len(error_chain) > 0:
        error_integrator = pyv.ErrIntegrator(
            error_chain,
            sens_data,
            tc_array.get_measurement_shape(),
        )
        tc_array.set_error_integrator(error_integrator)

    measurements = tc_array.get_measurements()
    print(f"\nMeasurements for sensor at top of block:\n{measurements[-1,0,:]}\n")


    (fig,_) = pyv.plot_time_traces(tc_array,field_key)
    plt.show()

    save_fig = True
    if save_fig:
        image_path = Path.cwd() / "example_output"
        image_file = image_path / "monoblock_thermal_traces_syserrs.png"
        if not image_path.is_dir():
            image_path.mkdir()

        fig.savefig(image_file, dpi=300, format="png", bbox_inches="tight")


if __name__ == "__main__":
    main()
