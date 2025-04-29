# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import PIL


class DICSpeckleGen:
    """
    Dataclass holding summary information for the speckle pattern
    """
    def __init__(self,
                 seed=None,
                 px_vertical: int=720,
                 px_horizontal: int=1280,
                 size_radius: int=3,
                 size_stddev: float=0.0,
                 loc_variance: float=0.6,
                 loc_spacing: int=7,
                 smooth: bool=True,
                 smooth_stddev: float=1.0,
                 gray_level: int=4096,
                 pattern_digitisation: bool=True):

        self.seed = seed
        self.px_vertical = px_vertical
        self.px_horizontal = px_horizontal
        self.size_radius = size_radius
        self.size_stddev = size_stddev
        self.loc_variance = loc_variance
        self.loc_spacing = loc_spacing
        self.smooth = smooth
        self.smooth_stddev = smooth_stddev
        self.gray_level = gray_level
        self.pattern_digitisation = pattern_digitisation
        self.array = None

        # ensure gray level is valid
        if self.gray_level not in {256, 4096, 65536}:
            raise ValueError("gray_level must be one of {256, 4096, 65536}")

        # generate pattern and store in memory upon calling class
        self.generate_array()



    def generate_array(self) -> None:

        """
        Generate a speckle pattern based on default or user provided paramters.

        Args:
            None

        Returns:
            np.array: 2D speckle pattern.
        """

        # intialise speckle pattern based on datatype
        pattern_dtype = np.int32 if self.pattern_digitisation else np.float64          
        self.array = np.zeros((self.px_vertical, self.px_horizontal), dtype=pattern_dtype)


        # set random seed
        np.random.seed(self.seed)

        # speckles per row/col
        nspeckles_x = self.px_vertical // self.loc_spacing
        nspeckles_y = self.px_horizontal // self.loc_spacing

        # total number of speckles
        nspeckles = nspeckles_x * nspeckles_y

        # uniformly spaced grid of speckles.
        grid_x_uniform, grid_y_uniform = self._create_flattened_grid(nspeckles_x, nspeckles_y)

        # apply random shift
        low  = -self.loc_variance * self.loc_spacing
        high =  self.loc_variance * self.loc_spacing
        grid_x = self._random_shift_grid(grid_x_uniform, low, high, nspeckles)
        grid_y = self._random_shift_grid(grid_y_uniform, low, high, nspeckles)


        # pull speckle size from a normal distribution
        radii = np.random.normal(self.size_radius, self.size_stddev, nspeckles).astype(int)

        # loop over all grid points and create a circle mask. Mask then applied to pattern array.
        for ii in range(0, nspeckles):
            x,y,mask = self._circle_mask(grid_x[ii], grid_y[ii], radii[ii])
            self.array[x[mask], y[mask]] = self.gray_level-1


        # apply smoothing
        if self.smooth is True:
            self.array = gaussian_filter(self.array, self.smooth_stddev).astype(pattern_dtype)

        return None


    def get_array(self) -> np.ndarray:

        return self.array



    def show(self) -> None:
        """
        Display pattern as an image using Matplotlib.

        Returns:
        None
        """
        plt.figure()
        plt.xlabel('Pixel')
        plt.ylabel('Pixel')
        plt.imshow(self.array,cmap='gray', vmin=0, vmax=self.gray_level-1)
        plt.colorbar()
        plt.show()

        return None


    def save(self,filename: str) -> None:
        """
        Save the speckle pattern array as an image with PIL package.
        Image can either be saved as 8bit or 16bit image.
        Image Saved in .tiff format.

        Args:
        filename   (str): name/location of output image

        Returns:
        None: Saves image to directory withuser specified details.
        """

        if self.gray_level == 256:
            image = PIL.Image.fromarray((self.array).astype(np.uint8))
        elif (self.gray_level == 4096) or (self.gray_level == 65536):
            image = PIL.Image.fromarray((self.array).astype(np.uint16))
        else:
            raise ValueError("gray_level must be one of {256, 4096, 65536}")


        image.save(filename, format="TIFF")

        return None




    def _create_flattened_grid(self,
                               nspeckles_x: int,
                               nspeckles_y: int) -> tuple[np.ndarray,np.ndarray]:
        """
        Return a flattened grid for speckle locations.
        Evenly spaced grid based on axis size and the no. speckles along axis.
        Args:
            nspeckles_x (int): Number of speckles along x-axis
            nspeckles_y (int): Number of speckles along y-axis

        Returns:
            tuple (np.array, np.array): speckle indexes for each axis.
        """

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.px_vertical-1,  nspeckles_x),
                                        np.linspace(0, self.px_horizontal-1, nspeckles_y))

        grid_flattened_x = grid_x.flatten()
        grid_flattened_y = grid_y.flatten()

        return grid_flattened_x, grid_flattened_y


    def _random_shift_grid(self,
                           grid: np.array,
                           low: int,
                           high: int,
                           nsamples: int) -> np.array:
        """
        Takes a uniformly spaced grid as input as applies a unifirm random shift to each position.
        Args:
            grid   (np.array): grid to apply shifts
            low         (int): lowest possible value returned from shift
            high        (int): high possible value returned from shift
            nsamples    (int): number of speckles for shift.

        Returns:
            np.array: a numpy array of updated speckle locations after applying a random shift.
        """

        rand_shift_size = np.random.uniform(low, high, nsamples).astype(int)
        updated_grid = grid.astype(int) + rand_shift_size

        return updated_grid





    def _circle_mask(self,
                     pos_x: int,
                     pos_y: int,
                     radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a circular mask centered at speckle location position with a given radius. 
        The mask is applied within image bounds.

        Args:
            pos_x       (int): The x-coordinate of the center of the circle.
            pos_y       (int): The y-coordinate of the center of the circle.
            radius      (int): The radius of the circle (in pixels).

        Returns:
            tuple: A tuple containing:
                - x (np.ndarray): The x-coordinates of mask region.
                - y (np.ndarray): The y-coordinates of mask region.
                - mask (np.ndarray): Bool array containing speckle area.
        """

        min_x = max(pos_x - radius, 0)
        min_y = max(pos_y - radius, 0)
        max_x = min(pos_x + radius + 1, self.px_vertical)
        max_y = min(pos_y + radius + 1, self.px_horizontal)

        # Generate mesh grid of possible (xx, yy) points
        x, y = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))

        # Update the pattern for points inside the circle's radius
        mask = (x - pos_x)**2 + (y - pos_y)**2 <= radius**2

        return x, y, mask
