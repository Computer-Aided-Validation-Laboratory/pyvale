from multiprocessing import cpu_count
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import bpy
from dev_camerablender import CameraData

class RenderEngine(Enum):
    """Different render engines on Blender
    """
    CYCLES = "CYCLES"
    EEVEE = "BLENDER_EEVEE_NEXT"
    WORKBENCH = "BLENDER_WORKBENCH"

@dataclass
class RenderData:
    samples: int | None = None
    engine: RenderEngine = RenderEngine.CYCLES
    max_bounces: int = 12



class Render:
    def __init__(self,
                 render_data: RenderData | None = None,
                 image_path: str | None = None,
                 output_path: str | None = None,
                 cam_data: CameraData | None = None):
        self.render_data = render_data
        self.image_path = image_path
        self.output_path = output_path
        self.cam_data = cam_data
        self.scene = bpy.data.scenes['Scene']

        if render_data is None:

    def render_parameters(self,
                          file_name: str,
                          cores: int):
        bpy.context.scene.render.engine = self.render_data.engine.value
        bpy.context.scene.render.image_settings.color_mode = 'BW'
        self.scene.render.resolution_x = self.cam_data.sensor_px[0]
        self.scene.render.resolution_y = self.cam_data.sensor_px[1]
        self.scene.render.filepath =  str(self.image_path / file_name)
        self.scene.render.threads_mode = 'FIXED'
        self.scene.render.threads = cores

        if self.render_data.engine == RenderEngine.CYCLES:
            self.scene.cycles.use_denoising = False # To make rendering faster
            bpy.context.scene.cycles.samples = self.render_data.samples
            bpy.context.scene.cycles.max_bounces = self.render_data.max_bounces
        elif self.render_data.engine == RenderEngine.EEVEE:
            self.scene.eevee.taa_render_samples = self.render_data.samples

        bpy.context.scene.render.image_settings.file_format = 'TIFF'

        bpy.ops.render.render(write_still=True)

    def render_image(self,
                     name: int,
                     image_count: int,
                     part,
                     cam_count: int | None = None):
        if cam_count is not None:
            file_name = name + '_' + str(image_count) + '_' + str(cam_count) + '.tiff'
        else:
            file_name = name + '_' + str(image_count) + '.tiff'
        cores = int(cpu_count())
        self.render_parameters(file_name, cores)
        self._write_progress(image_count, part)

    def _write_progress(self, image_count: int, part):
        if Path(self.output_path).is_dir() is False:
            Path.mkdir(self.output_path)
        self.output_path = self.output_path / 'output.txt'
        if image_count == 0:
            report = open(self.output_path, 'w', encoding='utf-8')
        else:
            report = open(self.output_path, 'a', encoding='utf-8')
        report.write('\nOn render: ' + str(image_count))
        report.write('\nPart dimensions: ' + str(part.dimensions))
        report.write('\nPart location: ' + str(part.location))
        report.write('\nPart rotation: ' + str(part.rotation_euler))
        report.write('\n')
        report.close()






