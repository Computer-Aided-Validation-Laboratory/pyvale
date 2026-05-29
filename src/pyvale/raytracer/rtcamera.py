# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

# NOTE: IF CAMERA VIEW BREAKS FOR YOU COMPARED TO THE PREVIOUS VERSIONS, CHECK THE BOTTOM OF _create_viewport AND UNCOMMENT THE PREVIOUS VERSION

import numpy as np
from math import tan, radians

# Utility function used across the code below
def normalise_vector(vector: np.ndarray) -> np.ndarray:
    '''Returns the normalised vector, i.e., with length 1.0.'''
    return vector / np.sqrt(vector.dot(vector))

class Camera:
    """
    Creates a camera and associated viewport.

    Default parameters have the camera at the world origin, looking straight at the viewport 1 world unit away.
    """

    __slots__ = ['image_width', 'image_height', 'camera_center', 'point_camera_target', 'angle_vertical_view', 'vector_view_up','matrix_camera_to_world',
                 'matrix_world_to_camera', 'matrix_rotation', 'matrix_pixel_spacing', 'viewport_upper_left', 'pixel_00_center', 'focal_length', 'angle_cone', 'matrix_defocus_disc']

    def __init__(self, image_width: int, image_height: int, camera_center: np.ndarray = np.array([0.0, 0.0, 0.0]), point_camera_target: np.ndarray = np.array([0, 0, -1]),
                 angle_vertical_view: float = 90.0, focal_length: float = 0.0, angle_cone: float = 0.0):
        # Defining camera parameters
        self.image_width = image_width
        self.image_height = image_height
        self.camera_center = camera_center
        self.point_camera_target = point_camera_target
        self.angle_vertical_view = radians(angle_vertical_view)  # Assume angle_vertical_view is in degrees, so convert to radians. It determines the FOV
        self.vector_view_up = np.array([0.0, 1.0, 0.0]) # View up vector orthogonal to basis_vector_right. Defines sideways tilt. Value can be changed, this is the default for the camera to be straight.
        # Parameters for Depth of Field (thin lens approximation)
        self.focal_length = focal_length # If left at 0.0, this is focal_length; else we have focus distance
        self.angle_cone = radians(angle_cone) # 0 => No blur?
        self.matrix_defocus_disc = np.zeros((2, 3), dtype=np.float64)
        # Basis matrices
        self.matrix_camera_to_world = np.zeros((4, 4), dtype=np.float64)
        self.matrix_world_to_camera = np.zeros((4, 4), dtype=np.float64)
        self.matrix_rotation = np.zeros((3, 3), dtype=np.float64) # = matrix_camera_to_world[:3, :3]
        # Viewport parameters
        self.matrix_pixel_spacing = np.zeros((2, 3), dtype=np.float64)
        self.viewport_upper_left = np.zeros((3), dtype=np.float64)
        self.pixel_00_center = np.zeros((3), dtype=np.float64)
        self.create_basis_matrices()

    def create_basis_matrices(self) -> None:
        """
        Creates camera-to-world matrix.
        """
        self.matrix_camera_to_world = np.zeros((4, 4))
        basis_vector_forward, basis_vector_right, basis_vector_up = self._compute_camera_basis_vectors()
        self.matrix_camera_to_world[:, :3] = np.array(
            [basis_vector_right, basis_vector_up, basis_vector_forward, self.camera_center])
        self.matrix_camera_to_world[3] = np.array([0.0, 0.0, 0.0, 1.0])
        self.matrix_world_to_camera = np.linalg.inv(self.matrix_camera_to_world)
        self.matrix_rotation = np.column_stack((basis_vector_right, basis_vector_up, basis_vector_forward))
        self._create_viewport(basis_vector_forward, basis_vector_right, basis_vector_up)
        self._find_defocus_basis_vectors(basis_vector_right, basis_vector_up)
        # return self.matrix_camera_to_world, self.matrix_world_to_camera

    def _compute_camera_basis_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates the camera basis vectors from the camera center and the point the camera is looking at.

        Parameters:
        -----------
        None
        
        Returns:
        --------
        tuple[np.ndarray, np.ndarray, np.ndarray, float]
            Normalised forward, right, and up vectors of the camera basis.
           xxx Focal length of the camera.
        """
        # camera_center = lookfrom, point_camera_target = lookat if we use the ScratchAPixel naming convention
        basis_vector_forward = self.camera_center - self.point_camera_target
        if self.focal_length == 0.0:
           self.focal_length = np.sqrt(basis_vector_forward.dot(basis_vector_forward))
        basis_vector_right = np.cross(self.vector_view_up, basis_vector_forward)
        basis_vector_up = np.cross(basis_vector_forward, basis_vector_right)
        return normalise_vector(basis_vector_forward), normalise_vector(basis_vector_right), normalise_vector(basis_vector_up)

    def _create_viewport(self,
                         basis_vector_forward: np.ndarray,
                         basis_vector_right: np.ndarray,
                         basis_vector_up: np.ndarray) -> None:
        """
        Creates the viewport from the camera basis vectors and the focal length.

        Sets the pixel spacing vectors and the 0,0-positions for the pixel and the upper left corner of the viewport.
        
        Parameters:
        -----------
        basis_vector_forward: np.ndarray
            The forward direction of the camera.
        basis_vector_right: np.ndarray
            The right direction of the camera.
        basis_vector_up: np.ndarray
            The up direction of the camera.
        
        Returns:
        --------
        None
        
        """
        focal_length = self.focal_length
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
        # Previous version - uncomment to compare to previous test cases. Had it with normalised basis vector forward for ages, which was wrong
        #self.viewport_upper_left = self.camera_center - basis_vector_forward - vector_viewport_x_axis / 2 - vector_viewport_y_axis / 2 
        # Corrected version - accounts for the focal lenght now
        self.viewport_upper_left = self.camera_center - (focal_length * basis_vector_forward) - vector_viewport_x_axis / 2 - vector_viewport_y_axis / 2
        self.pixel_00_center = self.viewport_upper_left + 0.5 * (vector_pixel_spacing_x + vector_pixel_spacing_y)

    def _find_defocus_basis_vectors(self, basis_vector_right, basis_vector_up):
        cone_tan = tan(self.angle_cone / 2)
        radius_defocus_disc = self.focal_length * cone_tan
        print(f"Defocus disc radius: {radius_defocus_disc}")
        # Double check if these should be unit vectors or not
        vector_horizontal_radius = basis_vector_right * radius_defocus_disc
        vector_vertical_radius = basis_vector_up * radius_defocus_disc
        self.matrix_defocus_disc[0] = vector_horizontal_radius
        self.matrix_defocus_disc[1] = vector_vertical_radius



    def calculate_view_dims(self) -> tuple[float, float, float]:
        """
        Calculates the dimensions for the given camera view: focal length, the height and width of the viewport, and the position of the bottom right corner.

        Parameters:
        -----------
        None
        
        Returns:
        --------
        None
        
        """
        h_temp = tan(self.angle_vertical_view / 2)
        viewport_height = 2 * h_temp * self.focal_length  # world units (arbitrary)
        viewport_width = viewport_height * (self.image_width / self.image_height)  # world units (arbitrary)
        viewport_bottom_right = np.array(
            [self.viewport_upper_left[0] + viewport_width, self.viewport_upper_left[1] - viewport_height,
                self.viewport_upper_left[2]])
        return viewport_height, viewport_width, viewport_bottom_right
    
    def print_view_dims(self) -> None:
        """
        Prints the relevant dimensions and positions for the given camera view: camera position, its target, viewport location, focal length, FOV, pixel spacing.
        
            Helper function to help position objects in the scene, as the scene is defined in world units, not pixels, and scales with the image size and vertical FOV.

        Parameters:
        -----------
        None
        
        Returns:
        --------
        None
        
        """
        viewport_height, viewport_width, viewport_bottom_right = self.calculate_view_dims()
        print(f"Camera position [world units]: {np.round(self.camera_center,3)}")
        print(f"Lookat position [world units]: {np.round(self.point_camera_target,3)}")
        print(f"Vertical FOV [degrees]: {np.round(np.degrees(self.angle_vertical_view),3)}")
        print(f"Focal length [world units]: {np.round(self.focal_length,2)}")
        print(f"Viewport size [world units]:\n \twidth: {np.round(viewport_width,3)}\n \theight: {np.round(viewport_height,)}")
        print(f"Viewport coordinates[world units]:\n \ttop left corner: {np.round(self.viewport_upper_left,3)}\n \tbottom right corner: {np.round(viewport_bottom_right,3)}")
        print(f"Pixel spacing [world units]:\n \thorizontal (left->right): {np.round(self.matrix_pixel_spacing[0,0],6)}\n \tvertical (top->bottom): {np.abs(np.round(self.matrix_pixel_spacing[1,1],6))}")