# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Pyvale example: TODO
--------------------------------------------------------------------------------
TODO

Test case: TODO
"""

from pathlib import Path
import numpy as np
import mooseherder as mh
import pyvale as pyv


def main() -> None:

    data_path = pyv.DataSet.thermal_3d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=None)
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    pyv.print_dimensions(sim_data)

    n_sens = (1,4,1)
    x_lims = (12.5,12.5)
    y_lims = (0,33.0)
    z_lims = (0.0,12.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sens_data = pyv.SensorData(positions=sens_pos)

    field_key = 'temperature'
    tc_array = pyv.SensorArrayFactory() \
        .thermocouples_basic_errs(sim_data,
                                  sens_data,
                                  field_key,
                                  elem_dims=3)

    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for sensor at top of block:\n{measurements[-1,0,:]}\n')

    vis_opts = pyv.VisOptsSimSensors()
    vis_opts.window_size_px = (1200,800)
    vis_opts.camera_position = np.array([(59.354, 43.428, 69.946),
                                         (-2.858, 13.189, 4.523),
                                         (-0.215, 0.948, -0.233)])

    vis_mode = "vector"
    save_dir = Path.cwd() / "exampleoutput"
    if not save_dir.is_dir():
        save_dir.mkdir()

    if vis_mode == "animate":
        anim_opts = pyv.VisOptsAnimation()

        anim_opts.save_path = save_dir / "test_animation"
        anim_opts.save_animation = pyv.EAnimationType.MP4

        pv_anim = pyv.animate_sim_with_sensors(tc_array,
                                                  field_key,
                                                  time_steps=None,
                                                  vis_opts=vis_opts,
                                                  anim_opts=anim_opts)

    else:
        image_save_opts = pyv.VisOptsImageSave()

        image_save_opts.path = save_dir / "test_vector_graphics"
        image_save_opts.image_type = pyv.EImageType.SVG

        pv_plot = pyv.plot_point_sensors_on_sim(tc_array,
                                                field_key,
                                                time_step=-1,
                                                vis_opts=vis_opts,
                                                image_save_opts=image_save_opts)
        pv_plot.show()


if __name__ == '__main__':
    main()
