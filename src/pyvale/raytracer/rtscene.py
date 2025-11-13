from dataclasses import dataclass, field
import numpy as np

@dataclass(slots=True)
class Scene:
    '''WIP: Dataclass for storing camera, mesh, and light data in a format that should work best with C++
    while preserving user-friendly interface.'''
    scene_connectivity: list[np.ndarray] = field(default_factory=list)
    scene_coords: list[np.ndarray] = field(default_factory=list)
    scene_face_colors: list[np.ndarray] = field(default_factory=list)
    scene_camera_center: list[np.ndarray] = field(default_factory=list)
    scene_pixel_00_center: list[np.ndarray] = field(default_factory=list)
    scene_matrix_pixel_spacing: list[np.ndarray] = field(default_factory=list)
        
    def add_mesh(self, connectivity:np.ndarray, coords: np.ndarray, face_colors: np.ndarray) -> None:
        '''Adds a mesh to the scene.'''
        self.scene_connectivity.append(connectivity)
        self.scene_coords.append(coords)
        self.scene_face_colors.append(face_colors)

    def add_camera (self, camera_center: np.ndarray, pixel_00_center: np.ndarray, matrix_pixel_spacing: np.ndarray) -> None:
        '''Adds a camera to the scene.'''
        self.scene_camera_center.append(camera_center)
        self.scene_pixel_00_center.append(pixel_00_center)
        self.scene_matrix_pixel_spacing.append(matrix_pixel_spacing)
 