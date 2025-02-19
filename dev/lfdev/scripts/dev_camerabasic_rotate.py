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
from scipy.spatial.transform import Rotation
import mooseherder as mh
import pyvale

def main() -> None:
    data_path = Path('src/pyvale/data/case17_out.e')
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    descriptor = pyvale.SensorDescriptorFactory.displacement_descriptor()

    spat_dims = 2
    field_key = 'disp'
    components = ('disp_x','disp_y')
    disp_field = pyvale.FieldVector(sim_data,field_key,components,spat_dims)


    num_px = np.array((250,500))
    leng_per_px = pyvale.CameraTools.calc_resolution_from_sim_2d(num_px,
                                                  sim_data.coords,
                                                  pixels_border=5)
    roi_center_world = pyvale.CameraTools.calc_roi_cent_from_sim_2d(sim_data.coords)
    sensor_angle = Rotation.from_euler("zyx", [180, 0, 0], degrees=True)

    cam_data_norot = pyvale.CameraData2D(pixels_count=num_px,
                                   leng_per_px=leng_per_px,
                                   roi_cent_world=roi_center_world)
    cam_data_rot = pyvale.CameraData2D(pixels_count=num_px,
                                leng_per_px=leng_per_px,
                                roi_cent_world=roi_center_world,
                                angle=sensor_angle)

    print(f"{cam_data_norot.roi_cent_world=}")

    camera = pyvale.CameraBasic2D(cam_data=cam_data_norot,
                                  field=disp_field,
                                  descriptor=descriptor)
    camera_rot = pyvale.CameraBasic2D(cam_data=cam_data_rot,
                                field=disp_field,
                                descriptor=descriptor)

    measurements = camera.calc_measurements()
    meas_images = camera.get_measurement_images()
    measurements_rot = camera_rot.calc_measurements()
    meas_images_rot = camera.get_measurement_images()

    print(80*"=")
    print(f"{measurements.shape=}")
    print(f"{meas_images.shape=}")
    print(80*"=")

    (fig,ax) = pyvale.plot_measurement_image(camera,"disp_x")
    ax.set_title("No rotation")
    (fig,ax) = pyvale.plot_measurement_image(camera_rot,"disp_x")
    ax.set_title("Rotated")
    plt.show()

if __name__ == "__main__":
    main()