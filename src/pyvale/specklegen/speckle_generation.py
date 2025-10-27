import numpy as np
from scipy import ndimage
from perlin_numpy import (
    generate_perlin_noise_2d, generate_fractal_noise_2d)
import opensimplex as simplex

def pixelsInDisk(cent_x: int, cent_y: int, 
                 screen_size_width: int, screen_size_height: int, 
                 speckle_size: float, foreground_colour: int, image: np.ndarray,
                 check_overlap: bool) -> int:
        
        """ Set pixels in a disk shape on the image array, disk centered at (cent_x, cent_y) with given speckle_size """
        "Inputs:"
        " cent_x, cent_y: center coordinates of the disk"
        " screen_size_width, screen_size_height: dimensions of the image"
        " speckle_size: diameter of the disk"
        " foreground_colour: colour value to set for the disk pixels"
        " image: 2D numpy array representing the image"
        " check_overlap: if True, check for overlap with existing foreground pixels and does not modify the image"
        " Output: 0 if overlap detected, 1 if no overlap (when check_overlap is True), otherwise modifies the image"

        box_max_x = int(np.ceil(cent_x + speckle_size / 2))
        box_min_x = int(np.floor(cent_x - speckle_size / 2))
        
        box_max_y = int(np.ceil(cent_y + speckle_size / 2))
        box_min_y = int(np.floor(cent_y - speckle_size / 2))

        if check_overlap:
            over_lap = False

            for yy in range(box_min_y, box_max_y):
                for xx in range(box_min_x, box_max_x):
                    if yy < 0 or yy >= screen_size_height or xx < 0 or xx >= screen_size_width:
                        continue
                    distance = (xx + 0.5 - cent_x)**2 + (yy + 0.5 - cent_y)**2
                    if distance <= (speckle_size / 2)**2:
                        if image[yy, xx] == foreground_colour:
                            over_lap = True
                            break
                if over_lap:
                    # print(f"Overlap detected at ({xx}, {yy})")
                    break
            return 0 if over_lap else 1
        
        else:
            for yy in range(box_min_y, box_max_y):
                for xx in range(box_min_x, box_max_x):
                    if yy < 0 or yy >= screen_size_height or xx < 0 or xx >= screen_size_width:
                        continue
                    distance = (xx + 0.5 - cent_x)**2 + (yy + 0.5 - cent_y)**2
                    if distance <= (speckle_size / 2)**2:
                        image[yy, xx] = foreground_colour


def generate_speckles(screen_size_width: int, screen_size_height: int, 
                      speckle_size: float, foreground_colour: int,
                      total_speckles: int, reduce_overlap: bool,
                      bit_depth: int, background_colour: int,
                      sigma: float, attempts_tot: int = 100) -> np.ndarray:
     
        """ Generate a speckle pattern image with given parameters """
        """ The speckles are disks of given speckle_size, placed randomly on the image, based on the uniform probability distribution """
        """ Inputs: """
        """ screen_size_width, screen_size_height: dimensions of the image """
        """ speckle_size: diameter of each speckle disk """
        """ foreground_colour: colour value for the speckles """
        """ total_speckles: total number of speckles to place """
        """ reduce_overlap: if True, attempts to reduce overlap between speckles """
        """ bit_depth: bit depth of the image (8 or 16) """
        """ background_colour: colour value for the background """
        """ sigma: standard deviation for Gaussian blur applied after speckle placement """
        """ attempts_tot: maximum number of attempts to place each speckle without overlap """
        """ Output: speckle pattern image as a 2D numpy array and speckle generation stats """

        image: np.ndarray = np.ones((screen_size_height,screen_size_width),
                                    dtype=np.uint16 if bit_depth == 16 else np.uint8) * background_colour
        
        results = np.zeros([total_speckles, 5])  # speckle number, attempts, with/without/not checked overlap (1/0/2), cent_x, cent_y

        if reduce_overlap:
                for i in range(int(total_speckles)):
                    # reduce overlap by checking if the new speckle overlaps with existing ones
                    check_overlap = True
                    over_lap = True
                    attempts = 0
                    # print(f"Placing speckle {i+1}/{total_speckles}")
                    while over_lap and attempts < attempts_tot:
                        # print(f"Attempt {attempts+1} to place speckle {i+1}")
                        over_lap = False
                        cent_x = np.random.uniform(0, screen_size_width)
                        cent_y = np.random.uniform(0, screen_size_height)
            
                        # check colour
                        if image[int(cent_y), int(cent_x)] == foreground_colour:
                            over_lap = True
                            attempts += 1
                            # print(f"Overlap detected at centre ({cent_x}, {cent_y})")
                            continue
                        over_lap_int = pixelsInDisk(cent_x, cent_y, 
                                screen_size_width, screen_size_height, 
                                speckle_size, foreground_colour, image, check_overlap)
                        over_lap = True if over_lap_int == 0 else False

                        results[i, 0] = i + 1
                        results[i, 1] = attempts + 1
                        results[i, 2] = 0 if over_lap else 1
                        results[i, 3] = cent_x
                        results[i, 4] = cent_y
                        attempts += 1
                    # if attempts == attempts_tot:
                    #     print("Could not place speckle without overlap, placing anyway.")
                    # Place the speckle finally
                    pixelsInDisk(cent_x, cent_y, 
                                screen_size_width, screen_size_height, 
                                speckle_size, foreground_colour, image, check_overlap=False)
        else:
            check_overlap = False
            for i in range(int(total_speckles)):
                cent_x = np.random.uniform(0, screen_size_width)
                cent_y = np.random.uniform(0, screen_size_height)
                pixelsInDisk(cent_x, cent_y, 
                        screen_size_width, screen_size_height, 
                        speckle_size, foreground_colour, image, check_overlap)
                results[i, 0] = i + 1
                results[i, 1] = 1
                results[i, 2] = 2
                results[i, 3] = cent_x
                results[i, 4] = cent_y
             
            
        # Apply Gaussian blur
        image_blur = np.copy(image)
        image_blur = ndimage.gaussian_filter(image_blur, sigma=sigma)
        
        return image_blur, results


def generate_speckles_perlin_noise(screen_size_width: int, screen_size_height: int,
                                   res_width: int, res_height: int,
                                   foreground_colour: int,
                                   bit_depth: int, background_colour: int, type_gen: str,
                                   **kwargs) -> np.ndarray:
    
    """This function uses noise implementation from perlin-numpy package 
    (https://github.com/pvigier/perlin-numpy/tree/master)"""

    """Generate a speckle pattern image using Perlin or fractal noise. """
    """ Inputs: """
    """ screen_size_width, screen_size_height: dimensions of the image """
    """ res_width, res_height: number of periods of noise to generate along 2 axes """
    """ foreground_colour: colour value for the speckles """
    """ bit_depth: bit depth of the image (8 or 16) """
    """ background_colour: colour value for the background """
    """ type_gen: type of noise to generate ('perlin' or 'fractal')"""
    """ kwargs: Additional keyword arguments for fractal noise generation (e.g., octaves)"""
    """ Output: speckle pattern image as a 2D numpy array """

    if type_gen == "perlin":
        # screen size must be a multiple of res
        assert screen_size_width % res_width == 0, "The screen size must be a multiple of res."
        assert screen_size_height % res_height == 0, "The screen size must be a multiple of res."
        image = generate_perlin_noise_2d(shape = (screen_size_height, screen_size_width), 
                                          res = (res_height, res_width))
        
    if type_gen == "fractal":
        # screen size must be a multiple of lacunarity^(octaves-1)*res
        assert screen_size_width % res_width == 0, "The screen size must be a multiple of lacunarity^(octaves-1)*res."
        assert screen_size_height % res_height == 0, "The screen size must be a multiple of lacunarity^(octaves-1)*res."
        octaves = kwargs["octaves"]
        image = generate_fractal_noise_2d(shape = (screen_size_height, screen_size_width), 
                                          res = (res_height, res_width),
                                          octaves = octaves)

    # scale to background and foreground colours
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val)  # Normalise to [0, 1]
    image = image * (foreground_colour - background_colour) + background_colour
    image = image.astype(np.uint16 if bit_depth == 16 else np.uint8)

    return image

def generate_speckles_simplex_noise(screen_size_width: int, screen_size_height: int,
                                   foreground_colour: int,
                                   bit_depth: int, background_colour: int,
                                   feature_size_width: float, feature_size_height: float,
                                   seed: int) -> np.ndarray:
     
    """This function uses noise implementation from opensimplex package
    (https://pypi.org/project/opensimplex/)
    (https://code.larus.se/lmas/opensimplex)"""

    """Generate a speckle pattern image using OpenSimplex noise (patent-free). """
    """ Inputs: """
    """ screen_size_width, screen_size_height: dimensions of the image """
    """ foreground_colour: colour value for the speckles """
    """ bit_depth: bit depth of the image (8 or 16) """
    """ background_colour: colour value for the background """
    """ feature_size_width: controls the size of features in the noise pattern (speckle size width-wise) """
    """ feature_size_height: controls the size of features in the noise pattern (speckle size height-wise) """
    """ seed: seed for the noise generation """
    """ Output: speckle pattern image as a 2D numpy array """

    simplex.seed(seed)

    # image: np.ndarray = np.zeros((screen_size_height,screen_size_width))

    # for y in range(0, screen_size_height):
    #     for x in range(0, screen_size_width):
    #         value = simplex.noise2(x/feature_size, y/feature_size)
    #         image[y, x] = value

    ix, iy = np.arange(screen_size_width), np.arange(screen_size_height)
    image = simplex.noise2array(ix/feature_size_width, iy/feature_size_height)

    # scale to background and foreground colours
    min_val = np.min(image)
    max_val = np.max(image)
    # print(f"Min and max values before scaling: {min_val}, {max_val}")
    image = (image - min_val) / (max_val - min_val)  # Normalise to [0, 1]
    image = image * (foreground_colour - background_colour) + background_colour
    image = image.astype(np.uint16 if bit_depth == 16 else np.uint8)

    return image
    
