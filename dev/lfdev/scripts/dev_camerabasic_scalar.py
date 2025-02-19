"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale

def main() -> None:
    data_path = Path('src/pyvale/data/case13_out.e')
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_key = list(sim_data.node_vars.keys())[0] # type: ignore
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    descriptor = pyvale.SensorDescriptorFactory.temperature_descriptor()

    field_key = 'temperature'
    t_field = pyvale.FieldScalar(sim_data,
                                 field_key=field_key,
                                 spat_dims=2)

    num_px = np.array((500,250))
    leng_per_px = pyvale.calc_resolution_from_sim_2d(num_px,
                                                  sim_data.coords,
                                                  pixels_border=5)
    roi_center_world = pyvale.calc_roi_cent_from_sim(sim_data.coords)

    cam_data = pyvale.CameraData2D(num_pixels=num_px,
                                   leng_per_px=leng_per_px,
                                   roi_center_world=roi_center_world)

    print(cam_data.roi_center_world)
    print(cam_data.roi_shift_world)


    camera = pyvale.CameraBasic2D(cam_data=cam_data,
                                  field=t_field,
                                  descriptor=descriptor)

    measurements = camera.calc_measurements()
    meas_images = camera.get_measurement_images()

    print(80*"=")
    print(f"{measurements.shape=}")
    print(f"{meas_images.shape=}")
    print(80*"=")

    (fig,ax) = pyvale.plot_measurement_image(camera,field_key)
    plt.show()

if __name__ == "__main__":
    main()