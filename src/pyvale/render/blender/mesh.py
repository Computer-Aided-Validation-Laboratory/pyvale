"""Blender compatibility conversion for historical pyvale simulation meshes."""

from collections.abc import Sequence

import numpy as np

from pyvale.dataio.simdata import SimData
from pyvale.sensorsim.enums import EDim
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_interp

from ..mesh import EElementType, Mesh3D


def mesh_from_simdata(
    sim_data: SimData,
    shader: object,
    displacement_keys: Sequence[str] | None = None,
    spatial_dimension: EDim = EDim.TWOD,
) -> Mesh3D:
    """Recreate legacy Blender surface extraction and deformation ordering."""
    keys = tuple(displacement_keys or ())
    surface = simdata_to_pyvista_interp(sim_data, keys or None, spatial_dimension)
    surface = surface.extract_surface(algorithm="dataset_surface")
    faces = np.asarray(surface.faces)
    nodes_per_face = int(faces[0])
    connectivity = faces.reshape((-1, nodes_per_face + 1))[:, 1:]
    displacements = None
    if keys:
        fields = [np.asarray(surface[key], dtype=np.float64) for key in keys]
        displacements = np.stack(fields, axis=2).transpose(1, 0, 2)
        if displacements.shape[2] == 2:
            displacements = np.pad(displacements, ((0, 0), (0, 0), (0, 1)))
    element_types = {3: EElementType.TRI3, 4: EElementType.QUAD4,
                     6: EElementType.TRI6, 8: EElementType.QUAD8,
                     9: EElementType.QUAD9}
    return Mesh3D(
        element_types[nodes_per_face], np.asarray(surface.points), connectivity,
        shader, displacements,
    )
