import bpy
import numpy as np
import time
from dataclasses import dataclass
from dev_partblender import PartBlender
from dev_render import RenderData, Render
from dev_camerablender import calc_FOV_mm

@dataclass
class CalibrationData:
    part: PartBlender | None = None,
    image_path: str | None = None,
    output_path: str | None = None,
    render_data: RenderData | None = RenderData
    cam_data: list | None = None
    angle_lims: tuple = (-10, 10)
    angle_step: int = 5
    plunge_lims: tuple = (-5, 5)
    plunge_step: int = 5


class Calibration():
    def __init__(self, cal_data):
        self.cal_data = cal_data

    def _get_FOV_mm(self):
        cam_data = self.cal_data.cam_data[0]
        FOV_mm = calc_FOV_mm(cam_data)
        return FOV_mm

    def calibrate(self):
        render_counter = 0
        for plunge in range(self.cal_data.plunge_lims[0],
                            self.cal_data.plunge_lims[1],
                            self.cal_data.plunge_step):
            # Plunge
            FOV_mm = self._get_FOV_mm()
            x_limit = int(round((FOV_mm[0] / 2) - (self.cal_data.part.dimensions[0] / 2)))
            y_limit = int(round((FOV_mm[1] / 2) - (self.cal_data.part.dimensions[1] / 2)))
            print('Plunge')

            for x in range(-x_limit, x_limit, 50):
                # Move in x-dir
                for y in range(-y_limit, y_limit, 50):
                    # Move in y-dir
                    self.cal_data.part.location = ((x, y, plunge))
                    print(f"{self.cal_data.part.location=}")
                    self.cal_data.part.location[2] = plunge
                    for angle in range(self.cal_data.angle_lims[0],
                            self.cal_data.angle_lims[1],
                            self.cal_data.angle_step):
                        # Rotate around x-axis
                        rotation  = (np.radians(angle), 0, 0)
                        self.cal_data.part.rotation_mode = 'XYZ'
                        self.cal_data.part.rotation_euler = rotation
                        for angle in range(self.cal_data.angle_lims[0],
                            self.cal_data.angle_lims[1],
                            self.cal_data.angle_step):
                            # Rotate around y-axis
                            rotation  = (0, np.radians(angle), 0)
                            self.cal_data.part.rotation_mode = 'XYZ'
                            self.cal_data.part.rotation_euler = rotation

                            self._render_cal_image(render_counter)
                            render_counter += 1
        print('Total number of calibration images = ' + str(render_counter))


    def _render_cal_image(self, render_counter):
        cam_count = 0
        render_name = 'cal_image'
        for cam in [obj for obj in bpy.data.objects if obj.type == 'CAMERA']:
            bpy.context.scene.camera = cam
            cam_data_render = self.cal_data.cam_data[cam_count]
            render = Render(self.cal_data.render_data,
                            image_path=self.cal_data.image_path,
                            output_path=self.cal_data.output_path,
                            cam_data=cam_data_render)

            render.render_image(render_name, render_counter, self.cal_data.part, cam_count)
            cam_count += 1


    def perform_calibration(self):
        render_start_time = time.perf_counter()
        self.calibrate()
        render_end_time = time.perf_counter()
        time_render = render_end_time - render_start_time
        print('Time taken to render images: ' + str(time_render) + 's')
        report = open((self.cal_data.output_path / 'output.txt'), 'a', encoding='utf-8')
        report.write('\nTime taken to render images: ' + str(time_render) + 's')
        report.close()









