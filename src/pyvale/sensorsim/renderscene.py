
#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
import numpy as np
from pyvale.sensorsim.cameradata import CameraData
from pyvale.sensorsim.rendermesh import RenderMesh

#===============================================================================
# TODO
# - How do we match render fields between meshes?
# - How do we check displacement fields are the same between meshes?
# - Eventually this will need to take render times and do the field interpolations
# - Check all render meshes to see if any are deformable


class RenderScene:
    __slots__ = ("cameras","meshes")

    def __init__(self,
                 cameras: list[CameraData] | None = None,
                 meshes: list[RenderMesh] | None = None,
                 ) -> None:
        if cameras is None:
            self.cameras = []
        else:
            self.cameras = cameras

        if meshes is None:
            self.meshes = []
        else:
            self.meshes = meshes

    def is_deformable(self) -> bool:
        for mm in self.meshes:
            if mm.fields_disp is not None:
                return True

        return False

def get_all_coords_world(meshes: list[RenderMesh]) -> np.ndarray:
    coords_all = []
    for mm in meshes:
        coords_all.append(np.matmul(mm.coords,mm.mesh_to_world_mat.T))

    return np.vstack(coords_all)



