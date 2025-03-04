import os
import numpy as np
from pathlib import Path
import bpy
from mooseherder.simdata import SimData
import pyvale

class PartBlender:
    """Creates an object in Blender
    """
    def __init__(self,
                 sim_data: SimData | None = None,
                 filename: str | None = None):
        self.sim_data = sim_data
        self.filename = filename

    def _get_spat_dim(self):
        nodes = self.sim_data.coords
        check_if_2d = np.count_nonzero(nodes, axis=0)
        if check_if_2d[2] == 0:
            spat_dim = 2
        else:
            spat_dim = 3
        return spat_dim

    def _get_components(self) -> tuple:
        node_vars = self.sim_data.node_vars
        node_vars_names = list(node_vars.keys())
        components = []
        if 'disp_x' in node_vars_names:
            components.append('disp_x')
        if 'disp_y' in node_vars_names:
            components.append('disp_y')
        if 'disp_z' in node_vars_names:
            components.append('disp_z')
        # if 'temperature' in node_vars_names:
        #     components.append('temperature')
        components = tuple(components)
        if len(components) == 0:
            components = None

        return components

    def _simdata_to_pvsurf(self, components, spat_dim, triangulate):
        self.sim_data.coords = centre_nodes(self.sim_data.coords * 1000)
        (pv_grid, pv_grid_vis) = pyvale.conv_simdata_to_pyvista(self.sim_data,
                                                                components,
                                                                spat_dim=spat_dim)

        pv_surf = pv_grid.extract_surface()
        if triangulate is True:
            tri_surf = pv_surf.triangulate()
            pv_surf = tri_surf

        return pv_surf, pv_grid


    def _pv_surf_to_obj(self, pv_surf):
        save_path = Path().cwd() / "test_output"
        if not save_path.is_dir():
            save_path.mkdir()
        name = "test_mesh.obj"
        save_file = save_path / name

        all_files = os.listdir(save_path)
        for ff in all_files:
            if name == ff:
                os.remove(save_path / ff)

        self.filename = str(save_file)
        pv_surf.save(save_file, binary=False)


    def import_from_obj(self, pv_surf = None):
        if self.filename is None:
            self._pv_surf_to_obj(pv_surf)

        bpy.ops.wm.obj_import(filepath=self.filename)

        part = bpy.context.selected_objects[0]
        return part

    def import_mesh(self, pv_surf, elements_per_face):
        vertices = pv_surf.points

        faces = pv_surf.faces.reshape(-1, elements_per_face)
        faces = np.delete(faces, 0, axis=1)
        print(f"{faces.shape=}")

        mesh = bpy.data.meshes.new('part')
        mesh.from_pydata(vertices, [], faces)
        part = bpy.data.objects.new('part', mesh)
        bpy.context.scene.collection.objects.link(part)

        return part

    def elements_per_face(self, pv_surf):
        elements_per_face = int((pv_surf.faces.shape[0] / pv_surf.n_cells))
        print(f"{pv_surf.faces.shape[0]=}")
        print(f"{pv_surf.n_cells=}")
        return elements_per_face


def centre_nodes(nodes):
        max = np.max(nodes, axis=0)
        min = np.min(nodes, axis=0)
        middle = max - ((max - min) / 2)
        centred = np.subtract(nodes, middle)
        return centred


