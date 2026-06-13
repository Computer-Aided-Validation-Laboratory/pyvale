import numpy as np
import vedo
import vtkmodules.all as vtk

# ================================================================================
# UV Alignment Interactor
# ================================================================================

class UVAlignmentInteractor(vtk.vtkInteractorStyleUser):
    """
    Custom VTK interactor for interactive UV alignment.

    Used in RTMesh align_uvs() method.

     Controls:
        LEFT-DRAG => translate
        RIGHT-DRAG => rotate (pivot = click point)
        MIDDLE-DRAG => scale (drag up=grow, down=shrink)
        SCROLL => zoom camera
        SHIFT+SCROLL => fine scale (±2% per tick)
        R => reset all transforms
        Q / Escape => confirm & close

    """

    def __init__(self, mesh_actor: vedo.Mesh,
                 original_verts: np.ndarray,
                 plt: vedo.Plotter,
                 texture_shape: tuple[int, int],
                 on_transform_update: callable):
        super().__init__()
        self._mesh = mesh_actor
        self._orig_verts = original_verts.copy()   # (N,3) pixel-space, unmodified
        self._plt = plt
        self._tex_h, self._tex_w = texture_shape[:2]
        self._callback = on_transform_update

        self._dragging = False
        self._rotating = False
        self._scaling = False

        self._last_position = None
        self._pivot_world = None

        # Accumulated transform components (for readout only)
        self._tot_translation = np.zeros(2)
        self._tot_angle_deg = 0.0
        self._tot_scale = 1.0

        # Register callbacks
        self.AddObserver("LeftButtonPressEvent", self._on_lbpress)
        self.AddObserver("LeftButtonReleaseEvent", self._on_lbrelease)
        self.AddObserver("RightButtonPressEvent", self._on_rbpress)
        self.AddObserver("RightButtonReleaseEvent", self._on_rbrelease)
        self.AddObserver("MiddleButtonPressEvent", self._on_mbpress)
        self.AddObserver("MiddleButtonReleaseEvent", self._on_mbrelease)
        self.AddObserver("MouseMoveEvent", self._on_move)
        self.AddObserver("MouseWheelForwardEvent", self._on_scroll_fwd)
        self.AddObserver("MouseWheelBackwardEvent", self._on_scroll_bwd)
        self.AddObserver("KeyPressEvent", self._on_key)

    # Helpers
    def _screen_to_world(self, sx, sy):
        ren = self._plt.renderer
        ren.SetDisplayPoint(sx, sy, 0)
        ren.DisplayToWorld()
        wx, wy, wz, ww = ren.GetWorldPoint()
        if ww != 0:
            wx, wy, wz = wx/ww, wy/ww, wz/ww
        return np.array([wx, wy])

    def _get_centroid(self):
        """World-space centroid of the current mesh position."""
        pts = np.array(self._mesh.vertices)
        return pts[:, :2].mean(axis=0)

    def _fire_callback(self):
        """Read back actual vertex positions → compute UV delta arrays."""
        current_verts = np.array(self._mesh.vertices)   # (N,3), pixel-space after transform
        self._callback(current_verts,
            self._orig_verts,
            self._tex_w, self._tex_h,
            self._tot_translation,
            self._tot_angle_deg,
            self._tot_scale,
            self._pivot_world)

    # Left mouse button
    def _on_lbpress(self, obj, event):
        sx, sy = self.GetInteractor().GetEventPosition()
        self._last_position = np.array([sx, sy], dtype=float)
        self._pivot_world = self._screen_to_world(sx, sy)
        self._dragging = True

    def _on_lbrelease(self, obj, event):
        self._dragging = False
        self._fire_callback()

    # Right mouse button
    def _on_rbpress(self, obj, event):
        sx, sy = self.GetInteractor().GetEventPosition()
        self._last_position = np.array([sx, sy], dtype=float)
        self._pivot_world = self._screen_to_world(sx, sy)
        self._rotating = True

    def _on_rbrelease(self, obj, event):
        self._rotating = False
        self._fire_callback()

    # Middle button
    def _on_mbpress(self, obj, event):
        sx, sy = self.GetInteractor().GetEventPosition()
        self._last_position = np.array([sx, sy], dtype=float)
        self._scaling = True

    def _on_mbrelease(self, obj, event):
        self._scaling = False
        self._fire_callback()

    # Mouse move
    def _on_move(self, obj, event):
        if not (self._dragging or self._rotating or self._scaling):
            return
        sx, sy  = self.GetInteractor().GetEventPosition()
        current = np.array([sx, sy], dtype=float)
        delta = current - self._last_position
        self._last_position = current

        pos = np.array(self._mesh.pos())

        if self._dragging:
            self._mesh.pos(pos[0] + delta[0], pos[1] + delta[1], 0)
            self._tot_translation += delta

        elif self._rotating:
            angle_deg = delta[0] * 0.5          # sensitivity: 0.5°/px
            self._tot_angle_deg += angle_deg
            px, py = self._pivot_world
            # Translate to pivot, rotate, translate back
            self._mesh.pos(pos[0] - px, pos[1] - py, 0)
            self._mesh.rotate_z(angle_deg)
            p2 = np.array(self._mesh.pos())
            self._mesh.pos(p2[0] + px, p2[1] + py, 0)

        elif self._scaling:
            # Vertical drag: up = scale up, down = scale down
            factor = 1.0 + delta[1] * 0.005    # sensitivity: 0.5% per px
            if factor <= 0:
                return
            self._tot_scale *= factor
            cx, cy = self._get_centroid()
            p = np.array(self._mesh.pos())
            # Move to centroid origin, scale, move back
            self._mesh.pos(p[0] - cx, p[1] - cy, 0)
            self._mesh.scale(factor)
            p2 = np.array(self._mesh.pos())
            self._mesh.pos(p2[0] + cx, p2[1] + cy, 0)

        self._plt.render()

    # Scroll
    def _on_scroll_fwd(self, obj, event):
        if self.GetInteractor().GetShiftKey():
            self._apply_scale(1.02)    # shift+scroll = fine scale
        else:
            self._plt.renderer.GetActiveCamera().Zoom(1.1)
            self._plt.render()

    def _on_scroll_bwd(self, obj, event):
        if self.GetInteractor().GetShiftKey():
            self._apply_scale(0.98)
        else:
            self._plt.renderer.GetActiveCamera().Zoom(0.9)
            self._plt.render()

    def _apply_scale(self, factor):
        self._tot_scale *= factor
        cx, cy = self._get_centroid()
        p = np.array(self._mesh.pos())
        self._mesh.pos(p[0] - cx, p[1] - cy, 0)
        self._mesh.scale(factor)
        p2 = np.array(self._mesh.pos())
        self._mesh.pos(p2[0] + cx, p2[1] + cy, 0)
        self._plt.render()

    # Key callbacks
    def _on_key(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == "r":
            # Hard-reset: restore original vertex positions directly
            self._mesh.vertices = self._orig_verts.copy()
            self._tot_translation[:] = 0
            self._tot_angle_deg = 0.0
            self._tot_scale = 1.0
            self._plt.renderer.ResetCamera()
            self._plt.render()
            print("[Reset] UV overlay restored to original positions.")
        elif key in ("q", "e", "Escape"):
            self._fire_callback()
            self._plt.close()

def get_transformed_uvs(current_verts: np.ndarray,
                        orig_verts: np.ndarray,
                        tex_w: int,
                        tex_h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Given the current pixel-space vertex positions (after interactive transform)
    and the original pixel-space positions, return new UV coordinates and the respective deltas.

    Parameters
    ----------
    current_verts : np.ndarray
        Shape (N,3), current pixel-space positions
    orig_verts : np.ndarray
        Shape (N,3), original pixel-space positions
    tex_w : int
        Texture width
    tex_h : int
        Texture height

    Returns
    -------
    new_uvs : np.ndarray
        Shape (N,2), UV coords in [0,1]^2  (Blender convention, v=0 bottom)
    delta_uvs : np.ndarray
        Shape (N,2), per-vertex UV delta from original location
    """
    new_uvs  = current_verts[:, :2] / np.array([tex_w, tex_h])
    orig_uvs = orig_verts[:, :2] / np.array([tex_w, tex_h])
    return new_uvs, new_uvs - orig_uvs
