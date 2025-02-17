import numpy as np
import bpy

class DeformMesh:
    def __init__(self,pv_surf, spat_dim, components):
        self.pv_surf = pv_surf
        self.spat_dim = spat_dim
        self.components = components

    def add_displacement(self, timestep: int, nodes):
        if set(self.components).issubset(self.pv_surf.array_names):
            added_disp = np.zeros_like(nodes)
            dim = 0
            for component in self.components:
                added_disp_1d = self.pv_surf.get_array(component)[:, timestep]
                added_disp[:, dim] = added_disp_1d * 1000
                dim += 1
            deformed_nodes = nodes + added_disp
            return deformed_nodes
        else:
            return None


class DeformPart:
    def __init__(self, part, deformed_nodes):
        self.part = part
        self.deformed_nodes = deformed_nodes

    def set_new_frame(self):
        frame_incr = 20
        ob = bpy.context.view_layer.objects.active
        if ob is None:
            bpy.context.objects.active = self.part

        current_frame = bpy.context.scene.frame_current
        current_frame += frame_incr
        bpy.context.scene.frame_set(current_frame)

        bpy.data.shape_keys['Key'].eval_time = current_frame
        self.part.data.shape_keys.keyframe_insert('eval_time', frame=current_frame)
        bpy.context.scene.frame_end = current_frame

    def deform_part(self):
        if self.part.data.shape_keys is None:
            self.part.shape_key_add()
            self.set_new_frame()
        sk = self.part.shape_key_add()
        self.part.data.shape_keys.use_relative = False

        n_nodes_layer = int(len(self.part.data.vertices))
        for i in range(len(self.part.data.vertices)):
            if i < n_nodes_layer:
                sk.data[i].co = self.deformed_nodes[i]


        return self.part







