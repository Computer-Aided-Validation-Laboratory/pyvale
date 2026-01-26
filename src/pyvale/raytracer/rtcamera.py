# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from email.mime import image
import numpy as np
from math import tan, radians
import pyvale.raytracer.rtscene

# Utility function used across the code below
def normalise_vector(vector: np.ndarray) -> np.ndarray:
    '''Returns the normalised vector, i.e., with length 1.0.'''
    return vector / np.sqrt(vector.dot(vector))

class Camera:
    '''Creates a camera and associated viewport.
    Default parameters have the camera at the world origin, looking straight at the viewport 1 world unit away.'''

    __slots__ = ['image_width', 'image_height', 'camera_center', 'point_camera_target', 'angle_vertical_view', 'matrix_camera_to_world',
                 'matrix_world_to_camera', 'matrix_pixel_spacing', 'viewport_upper_left', 'pixel_00_center']

    def __init__(self, image_width: int, image_height: int, camera_center: np.ndarray = np.array([0.0, 0.0, 0.0]), point_camera_target: np.ndarray = np.array([0, 0, -1]),
                 angle_vertical_view: float = 90.0):
        self.image_width = image_width
        self.image_height = image_height
        self.camera_center = camera_center
        self.point_camera_target = point_camera_target
        self.angle_vertical_view = radians(angle_vertical_view)  # Assume angle_vertical_view is in degrees, so convert to radians. It determines the FOV

        self.matrix_camera_to_world = np.zeros((4, 4), dtype=np.float64)
        self.matrix_world_to_camera = np.zeros((4, 4), dtype=np.float64)
        self.matrix_pixel_spacing = np.zeros((2, 3), dtype=np.float64)
        self.viewport_upper_left = np.zeros((3), dtype=np.float64)
        self.pixel_00_center = np.zeros((3), dtype=np.float64)
        self.create_basis_matrices()

    def create_basis_matrices(self) -> None:
        '''
        Creates camera-to-world matrix.
        '''
        self.matrix_camera_to_world = np.zeros((4, 4))
        basis_vector_forward, basis_vector_right, basis_vector_up, focal_length = self._compute_camera_basis_vectors()
        self.matrix_camera_to_world[:, :3] = np.array(
            [basis_vector_right, basis_vector_up, basis_vector_forward, self.camera_center])
        self.matrix_camera_to_world[3] = np.array([0.0, 0.0, 0.0, 1.0])
        self.matrix_world_to_camera = np.linalg.inv(self.matrix_camera_to_world)
        self._create_viewport(basis_vector_forward, basis_vector_right, basis_vector_up, focal_length)
        # return self.matrix_camera_to_world, self.matrix_world_to_camera

    def _compute_camera_basis_vectors(self):  # camera_center = lookfrom, point_camera_target = lookat
        '''Creates the camera basis vectors from the camera center and the point the camera is looking at.'''
        basis_vector_forward = self.camera_center - self.point_camera_target
        focal_length = np.sqrt(basis_vector_forward.dot(basis_vector_forward))
        vector_view_up = np.array([0.0, 1.0,
                                   0.0])  # View up vector orthogonal to basis_vector_right. Defines sideways tilt. Value can be changed, this is the default for the camera to be straight.
        basis_vector_right = np.cross(vector_view_up, basis_vector_forward)
        basis_vector_up = np.cross(basis_vector_forward, basis_vector_right)
        return normalise_vector(basis_vector_forward), normalise_vector(basis_vector_right), normalise_vector(
            basis_vector_up), focal_length

    def _create_viewport(self, basis_vector_forward: np.ndarray, basis_vector_right: np.ndarray, basis_vector_up: np.ndarray, focal_length: float) -> None:
        '''Creates the viewport from the camera basis vectors and the focal length.
        Returns pixel spacing vectors and the 0,0-positions for the pixel and the upper left corner of the viewport.'''
        h_temp = tan(self.angle_vertical_view / 2)
        viewport_height = 2 * h_temp * focal_length  # world units (arbitrary)
        viewport_width = viewport_height * (self.image_width / self.image_height)  # world units (arbitrary)
        # Viewport basis vectors
        vector_viewport_x_axis = viewport_width * basis_vector_right  # Vu
        vector_viewport_y_axis = (-viewport_height) * basis_vector_up  # Vw
        # Pixel spacing vectors (delta vectors)
        vector_pixel_spacing_x = vector_viewport_x_axis / self.image_width  # Delta u
        vector_pixel_spacing_y = vector_viewport_y_axis / self.image_height  # Delta v
        self.matrix_pixel_spacing = np.array([vector_pixel_spacing_x, vector_pixel_spacing_y])  # Store as an array
        # 0,0-positions
        self.viewport_upper_left = self.camera_center - basis_vector_forward - vector_viewport_x_axis / 2 - vector_viewport_y_axis / 2
        self.pixel_00_center = self.viewport_upper_left + 0.5 * (vector_pixel_spacing_x + vector_pixel_spacing_y)

    def add_camera_to_scene(self, scene) -> None:
        '''Adds the camera to the scene dataclass.'''
        scene.add_camera(self.camera_center, self.pixel_00_center, self.matrix_pixel_spacing)