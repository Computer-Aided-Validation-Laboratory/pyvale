# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: this module is a feature under developement.
"""

from pathlib import Path
from multiprocessing.pool import Pool
import numpy as np
from pyvale.cameradata import CameraData
from pyvale.rendermesh import RenderMeshData
import pyvale.cython.rastercyth as rastercyth

class RasterCY:
    @staticmethod
    def raster_static_mesh(cam_data: CameraData,
                           render_mesh: RenderMeshData,
                           threads_num: int | None = None,
                           ) -> tuple[np.ndarray,np.ndarray,np.ndarray] | None:

        frames_num = render_mesh.fields_render.shape[1]
        field_num = render_mesh.fields_render.shape[2]

        images_out = np.empty((cam_data.pixels_num[1],
                            cam_data.pixels_num[0],
                            frames_num,
                            field_num))
        depths_out = np.empty((cam_data.pixels_num[1],
                            cam_data.pixels_num[0],
                            frames_num))
        elems_out = np.empty((frames_num,))

        if threads_num is None:
            for tt in range(frames_num):
                (image_buffer,
                depth_buffer,
                elems_in_image) = rastercyth.raster_frame(
                                            render_mesh.coords,
                                            render_mesh.connectivity,
                                            render_mesh.fields_render[:,tt,:],
                                            cam_data)

                images_out[:,:,tt,:] = image_buffer
                depths_out[:,:,tt] = depth_buffer
                elems_out[tt] = elems_in_image

        else:
            assert threads_num > 0, "Number of threads must be greater than 0."

            with Pool(threads_num) as pool:
                processes_with_id = []

                for tt in range(frames_num):
                    args = (render_mesh.coords,
                            render_mesh.connectivity,
                            render_mesh.fields_render[:,tt,:],
                            cam_data)

                    process = pool.apply_async(rastercyth.raster_frame, args=args)
                    processes_with_id.append({"process": process,
                                              "frame": tt})

                for pp in processes_with_id:
                    (image,depth,elems) = pp["process"].get()
                    images_out[:,:,pp["frame"],:] = image
                    depths_out[:,:,pp["frame"]] = depth
                    elems_out = elems

        return (images_out,depths_out,elems_out)




