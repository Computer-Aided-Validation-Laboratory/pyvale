import numpy as np
from scipy import ndimage
import enum
from perlin_numpy import (
    generate_perlin_noise_2d, generate_fractal_noise_2d)
import opensimplex as simplex

class Theme(str, enum.Enum):
    BLACK_ON_WHITE = "black_on_white"
    WHITE_ON_BLACK = "white_on_black"

DTYPE_MAP = {
    8:  np.uint8,
    16: np.uint16,
    32: np.uint32
}

def pixelsInDisk(cent_x: int, cent_y: int, 
                 screen_size_width: int, screen_size_height: int, 
                 speckle_size: float, foreground_colour: int, image: np.ndarray,
                 check_overlap: bool) -> int:
        """A function to set pixels in a disk shape on the image array
    
        Parameters
        ----------
        cent_x : int
            Center x coordinate of the disk
        cent_y : int
            Center y coordinate of the disk
        screen_size_width : int
            Dimension of the image width-wise (pixels)
        screen_size_height : int
            Dimension of the image height-wise (pixels)
        speckle_size : float
            Diameter of the speckle disk (pixels)
        foreground_colour : int
           Colour value to set for the speckle pixels
        image : np.ndarray, shape=(num_px_y, num_px)
            2D numpy array representing the image
        check_overlap : bool
            If True, check for overlap with existing foreground pixels and does not modify the image
            If False, sets the pixels in the disk shape to the foreground colour
    
        Returns
        -------
        int
           0 if overlap detected, 1 if no overlap (when check_overlap is True), otherwise a function modifies the image
        """        
    
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

def postprocess_speckles(image: np.ndarray, seed: int, 
                         sigma_blur: float | None = None,
                         sigma_noise: float | None = None,
                         centre: int | None = None, contrast: int | None = None) -> np.ndarray:
    """A function to postprocess speckle pattern: re-position the irradiance distribution 
    and/or add a Gaussian blur and/or Gaussian noise.

    Parameters
    ----------
    image : np.ndarray, shape=(num_px_y, num_px)
        2D numpy array representing the speckle pattern
    seed : int
        Random seed for the noise generation
    sigma_blur : float, optional
        Standard deviation for Gaussian blur applied after speckle placement
    sigma_noise : float, optional
        Standard deviation for Gaussian noise applied after speckle placement
    centre : int, optional
        The centre of irradiance distribution
    contrast : int, optional
        The image contrast
    
    Returns
    -------
    image : np.ndarray, shape=(num_px_y, num_px)
        2D numpy array representing the postprocessed speckle pattern
    """

    np.random.seed(seed)

    # Apply Gaussian blur
    if sigma_blur is not None:
        image = ndimage.gaussian_filter(image, sigma=sigma_blur)

    # Apply Gaussian noise
    if sigma_noise is not None:
        noise = np.random.normal(0.0, sigma_noise, image.shape)
        image = image + noise

        # Re-position the irradiance distribution
    if centre and contrast is not None:
        v_max = centre + (contrast / 2)
        v_min = centre - (contrast / 2)
        image_min = np.min(image)
        image_max = np.max(image)
        image = (image - image_min) / (image_max - image_min)
        image = image * (v_max - v_min) + v_min

    return image

def get_colours(image_depth: int, container_depth: int, mode: str, theme) -> tuple[int,int]:

    """A function to calculate background and foreground colours.

    Parameters
    ----------
    image_depth : int
        Bit depth of the image/sensor (8, 10, 12, 16, or 32)
    container_depth : int
        Bit depth of the image container (8, 16, or 32)
    mode : str
        Mode to save image ("lower", "upper", "scaled")
    theme : Theme
        Black background with white speckles or reverse
    
    Returns
    -------
    tuple[int,int]
        Colour values for the background and speckles.
    """

    assert mode in ["lower", "upper", "scaled"], "Mode should be either lower, upper, or scaled."
    assert image_depth in [8, 10, 12, 16, 32], "Image bit depth should be either 8, 10, 12, 16, or 32."
    assert container_depth in [8, 16, 32], "Container bit depth should be either 8, 16, or 32."
    assert image_depth <= container_depth, \
    f"Image depth ({image_depth}) cannot exceed container depth ({container_depth})."
    
    if container_depth == image_depth:
        range_min = 0
        range_max = (2**container_depth) - 1

    else:
        if mode == "lower":
            # Use the lower bits of a container (e.g. 0 to 4095 for 12-bit)
            range_min = 0
            range_max = (2**image_depth) - 1
            
        elif mode == "upper":
            # Use the upper bits of a container (e.g. 61440 to 65535 for 12-bit)
            shift = container_depth - image_depth
            range_min = ((2**image_depth) - 1) << shift
            range_max = (2**container_depth) - 1 
            
        elif mode == "scaled":
            # Scale to fill the whole container (e.g. 0 to 65535 for 12-bit)
            range_min = 0
            range_max = (2**container_depth) - 1

    # Assign based on theme
    if theme == Theme.WHITE_ON_BLACK:
        background_colour = range_min
        foreground_colour = range_max
    else:
        background_colour = range_max
        foreground_colour = range_min
    
    return background_colour, foreground_colour

def generate_speckles_random_disks(screen_size_width: int, screen_size_height: int, 
                                   feature_size_width: float, feature_size_height: float,
                                   foreground_colour: int, background_colour: int,
                                   container_depth: int, type_gen: str, seed: int, **kwargs) -> np.ndarray:
        """A function to generate a speckle pattern image by randomly placing disk-shaped speckles
        based on uniform probability distribution.
    
        Parameters
        ----------
        screen_size_width : int
            Dimension of the image width-wise (pixels)
        screen_size_height : int
            Dimension of the image height-wise (pixels)
        feature_size_width : float
            Speckle size width-wise
        feature_size_height : float
            Speckle size height-wise
        foreground_colour : int
            Colour value for the speckles
        background_colour : int
            Colour value for the background
        container_depth : int
            Bit depth of the image container (8 or 16)
        type_gen : str
            Type of noise to generate (must be 'random_disks' for this function)
        seed : int
            Random seed for the noise generation
        total_speckles : int
            Total number of speckles to place
        reduce_overlap : bool
            If True, attempts to reduce overlap between speckles
        attempts_tot : int (optional)
            Maximum number of attempts to place each speckle without overlap (if reduce_overlap is True)
    
        Returns
        -------
        np.ndarray, shape=(num_px_y, num_px)
            Speckle pattern image as a 2D numpy array
        np.ndarray, shape=(total_speckles, 5)
            Speckle generation statistics
        """      

        speckle_size = (feature_size_width + feature_size_height) / 2
        total_speckles = kwargs.get("total_speckles")
        reduce_overlap = kwargs.get("reduce_overlap")
        attempts_tot = kwargs.get("attempts_tot", 100)

        # check that non of the required parameters are None
        assert speckle_size is not None, "speckle_size parameter is required."
        assert total_speckles is not None, "total_speckles parameter is required."
        assert reduce_overlap is not None, "reduce_overlap parameter is required."
        assert type_gen == "random_disks", "Type_gen must be 'random_disks' for this function."
        assert container_depth in [8, 16, 32], "Container bit depth should be either 8, 16, or 32."

        np.random.seed(seed)
        selected_dtype = DTYPE_MAP.get(container_depth)
        image: np.ndarray = np.ones((screen_size_height,screen_size_width),
                                    dtype=selected_dtype) * background_colour

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
        
        return image, results


def generate_speckles_random_disks_grid(screen_size_width: int, screen_size_height: int, 
                                   feature_size_width: float, feature_size_height: float,
                                   foreground_colour: int, background_colour: int,
                                   container_depth: int, type_gen: str, seed: int, **kwargs) -> np.ndarray:
        """A function to generate a speckle pattern image by randomly perturbating a grid of regularly-placed
        disk-shaped speckles based on discrete uniform probability distribution.
    
        Parameters
        ----------
        screen_size_width : int
            Dimension of the image width-wise (pixels)
        screen_size_height : int
            Dimension of the image height-wise (pixels)
        feature_size_width : float
            Speckle size width-wise
        feature_size_height : float
            Speckle size height-wise
        foreground_colour : int
            Colour value for the speckles
        background_colour : int
            Colour value for the background
        container_depth : int
            Bit depth of the image container (8 or 16)
        type_gen : str
            Type of noise to generate (must be 'random_disks_grid' for this function)
        seed : int
            Random seed for the noise generation
        total_speckles : int
            Total number of speckles to place
        perturbation_max : float
            Maximum amount to move speckles by during grid perturbation
    
        Returns
        -------
        np.ndarray, shape=(num_px_y, num_px)
            Speckle pattern image as a 2D numpy array
        np.ndarray, shape=(total_speckles, 5)
            Speckle generation statistics
        """        
    
        speckle_size = (feature_size_width + feature_size_height) / 2
        total_speckles = kwargs.get("total_speckles")
        perturbation_max = kwargs.get("perturbation_max")

        # check that none of the required parameters are None
        assert speckle_size is not None, "speckle_size parameter is required."
        assert total_speckles is not None, "total_speckles parameter is required."
        assert type_gen == "random_disks_grid", "Type_gen must be 'random_disks_grid' for this function."
        assert container_depth in [8, 16, 32], "Container bit depth should be either 8, 16, or 32."

        np.random.seed(seed)
        selected_dtype = DTYPE_MAP.get(container_depth)
        image: np.ndarray = np.ones((screen_size_height,screen_size_width),
                                    dtype=selected_dtype) * background_colour
        
        results = np.zeros([total_speckles, 5])  # speckle number, attempts, with/without/not checked overlap (1/0/2), cent_x, cent_y
        
        # Calculate the size of each grid cell and create initial speckle positions
        grid_cols = int(np.sqrt(total_speckles))
        grid_rows = int(np.ceil(total_speckles / grid_cols))
        grid_cell_width = screen_size_width // grid_cols
        grid_cell_height = screen_size_height // grid_rows

        i = np.arange(grid_rows)
        j = np.arange(grid_cols)
        speckle_positions_x = j * grid_cell_width + grid_cell_width // 2 + speckle_size // 2
        speckle_positions_y = i * grid_cell_height + grid_cell_height // 2 + speckle_size // 2
        speckle_positions = np.array(np.meshgrid(speckle_positions_x, speckle_positions_y)).T.reshape(-1, 2)

        # Remove extra speckle positions randomly
        total_speckles_diff = len(speckle_positions) - total_speckles
        
        if total_speckles_diff > 0:
            remove_indices = np.random.choice(len(speckle_positions), total_speckles_diff, replace=False)
            mask = np.ones(len(speckle_positions), dtype=bool)
            mask[remove_indices] = False
            speckle_positions = speckle_positions[mask]
                       
        # Perturb each speckle's position
        perturbations = np.random.randint(-perturbation_max, perturbation_max, size=(total_speckles, 2))
        perturbed_positions = []
        check_overlap = False
        for idx, (x, y) in enumerate(speckle_positions):
            # Add random perturbation to the x and y coordinates
            x_offset, y_offset = perturbations[idx]
            new_x = np.clip(x + x_offset, 0, screen_size_width - 1)
            new_y = np.clip(y + y_offset, 0, screen_size_height - 1)
            perturbed_positions.append((new_x, new_y))

            pixelsInDisk(new_x, new_y, 
                         screen_size_width, screen_size_height, 
                         speckle_size, foreground_colour, image, check_overlap)
            
            results[idx, 0] = idx + 1
            results[idx, 1] = 1
            results[idx, 2] = 2
            results[idx, 3] = new_x
            results[idx, 4] = new_y
                    
        return image, results

def generate_speckles_perlin_noise(screen_size_width: int, screen_size_height: int, 
                                   feature_size_width: float, feature_size_height: float,
                                   foreground_colour: int, background_colour: int,
                                   container_depth: int, type_gen: str, seed: int, **kwargs) -> np.ndarray:
    """A function to generate a speckle pattern image using Perlin noise or fractal noise.
    This function uses noise implementation from perlin-numpy package
    (https://github.com/pvigier/perlin-numpy/tree/master)

    Parameters
    ----------
    screen_size_width : int
        Dimension of the image width-wise (pixels)
    screen_size_height : int
        Dimension of the image height-wise (pixels)
    feature_size_width : float
        Speckle size width-wise
    feature_size_height : float
        Speckle size height-wise
    foreground_colour : int
        Colour value for the speckles
    background_colour : int
        Colour value for the background
    container_depth : int
        Bit depth of the image container (8 or 16)
    type_gen : str
        Type of noise to generate (must be 'perlin' or 'fractal' for this function)
    seed : int
        Random seed for the noise generation
    lacunarity : float (optional)
        Lacunarity parameter for fractal noise (required if type_gen is 'fractal')
    octaves : int (optional)
        Number of octaves for fractal noise (required if type_gen is 'fractal')

    Returns
    -------
    np.ndarray, shape=(num_px_y, num_px)
        Speckle pattern image as a 2D numpy array
    """    

    res_width = int(screen_size_width / feature_size_width)
    res_height = int(screen_size_height / feature_size_height)

    np.random.seed(seed)

    assert type_gen in ["perlin", "fractal"], "type_gen must be either 'perlin' or 'fractal' for this function."
    assert container_depth in [8, 16, 32], "Container bit depth should be either 8, 16, or 32."

    if type_gen == "perlin":
        assert screen_size_width % res_width == 0, "The screen width must be a multiple of res_width. res_width = int(screen_size_width / feature_size_width)."
        assert screen_size_height % res_height == 0, "The screen height must be a multiple of res_height. res_height = int(screen_size_height / feature_size_height)."
        image = generate_perlin_noise_2d(shape = (screen_size_height, screen_size_width), 
                                          res = (res_height, res_width))
        
    elif type_gen == "fractal":
        lacunarity = kwargs.get("lacunarity", None)
        octaves = kwargs.get("octaves", None)
        assert lacunarity is not None, "lacunarity parameter is required for fractal noise."
        assert octaves is not None, "octaves parameter is required for fractal noise."
        assert screen_size_width % (lacunarity ** (octaves - 1) * res_width) == 0, "The screen width must be a multiple of lacunarity^(octaves-1)*res_width. res_width = int(screen_size_width / feature_size_width)."
        assert screen_size_height % (lacunarity ** (octaves - 1) * res_height) == 0, "The screen height must be a multiple of lacunarity^(octaves-1)*res_height. res_height = int(screen_size_height / feature_size_height)."
        image = generate_fractal_noise_2d(shape = (screen_size_height, screen_size_width), 
                                          res = (res_height, res_width),
                                          octaves = octaves)

    # scale to background and foreground colours
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val)  # Normalise to [0, 1]

    if foreground_colour >= background_colour:
        # Black speckles on white background
        image = image * (foreground_colour - background_colour) + background_colour
    else:
        # White speckles on black background
        image = (1 - image) * (background_colour - foreground_colour) + foreground_colour

    
    selected_dtype = DTYPE_MAP.get(container_depth)
    image = image.astype(selected_dtype)

    return image

def generate_speckles_simplex_noise(screen_size_width: int, screen_size_height: int, 
                                   feature_size_width: float, feature_size_height: float,
                                   foreground_colour: int, background_colour: int,
                                   container_depth: int, type_gen: str, seed: int, **kwargs) -> np.ndarray:
    """A function to generate a speckle pattern image using Simplex noise.
    This function uses noise implementation from opensimplex package
    (https://pypi.org/project/opensimplex/)
    (https://code.larus.se/lmas/opensimplex)

    Parameters
    ----------
    screen_size_width : int
        Dimension of the image width-wise (pixels)
    screen_size_height : int
        Dimension of the image height-wise (pixels)
    feature_size_width : float
        Speckle size width-wise
    feature_size_height : float
        Speckle size height-wise
    foreground_colour : int
        Colour value for the speckles
    background_colour : int
        Colour value for the background
    container_depth : int
        Bit depth of the image container (8 or 16)
    type_gen : str
        Type of noise to generate (must be 'simplex' for this function)
    seed : int
        Random seed for the noise generation

    Returns
    -------
    np.ndarray, shape=(num_px_y, num_px)
        Speckle pattern image as a 2D numpy array
    """    

    assert type_gen == "simplex", "Type_gen must be 'simplex' for this function."
    assert container_depth in [8, 16, 32], "Container bit depth should be either 8, 16, or 32."
    simplex.seed(seed)

    ix, iy = np.arange(screen_size_width), np.arange(screen_size_height)
    image = simplex.noise2array(ix/feature_size_width, iy/feature_size_height)

    # scale to background and foreground colours
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val)  # Normalise to [0, 1]

    if foreground_colour >= background_colour:
        # Black speckles on white background
        image = image * (foreground_colour - background_colour) + background_colour
    else:
        # White speckles on black background
        image = (1 - image) * (background_colour - foreground_colour) + foreground_colour

    selected_dtype = DTYPE_MAP.get(container_depth)
    image = image.astype(selected_dtype)

    return image

def generate_speckles(screen_size_width: int, screen_size_height: int, 
                      feature_size_width: float, feature_size_height: float,
                      theme: Theme,
                      image_depth: int, container_depth: int, mode: str,
                      type_gen: str, 
                      seed: int, **kwargs) -> np.ndarray:
    """A function to generate a speckle pattern image using specified generation method.
    Speckle generation methods include 'random_disks', "random_disks_grid", 'perlin', 'fractal', and 'simplex'.

    Parameters
    ----------
    screen_size_width : int
        Dimension of the image width-wise (pixels)
    screen_size_height : int
        Dimension of the image height-wise (pixels)
    feature_size_width : float
        Speckle size width-wise
    feature_size_height : float
        Speckle size height-wise
    theme : Theme
        Black background with white speckles or reverse
    foreground_colour : int
        Colour value for the speckles
    background_colour : int
        Colour value for the background
    image_depth : int
        Bit depth of the image/sensor (8, 10, 12, 16, or 32)
    container_depth : int
        Bit depth of the image container (8, 16, or 32)
    mode : str
        Mode to fill image bits in container bits ("lower", "upper", "scaled")
    type_gen : str
        Type of noise to generate (must be one of 'random_disks', "random_disks_grid", 'perlin', 'fractal', 'simplex')
    seed : int
        Random seed for the noise generation
    sigma_blur : float, optional
        Standard deviation for Gaussian blur applied after speckle placement
    sigma_noise : float, optional
        Standard deviation for Gaussian noise applied after speckle placement
    centre : int, optional
        The centre of irradiance distribution
    contrast : int, optional
        The image contrast

    Returns
    -------
    np.ndarray, shape=(num_px_y, num_px)
        Speckle pattern image as a 2D numpy array
    np.ndarray, shape=(total_speckles, 5) (optional, if applicable)
        Speckle generation statistics

    Raises
    ------
    ValueError
        Unknown speckle generation type
    """    

    sigma_blur = kwargs.get("sigma_blur", None)
    sigma_noise = kwargs.get("sigma_noise", None)
    centre = kwargs.get("centre", None)
    contrast = kwargs.get("contrast", None)

    assert mode in ["lower", "upper", "scaled"], "Mode should be either lower, upper, or scaled."
    assert image_depth in [8, 10, 12, 16, 32], "Image bit depth should be either 8, 10, 12, 16, or 32."
    assert container_depth in [8, 16, 32], "Container bit depth should be either 8, 16, or 32."
    assert image_depth <= container_depth, \
    f"Image depth ({image_depth}) cannot exceed container depth ({container_depth})."
    
    background_colour, foreground_colour = get_colours(image_depth, container_depth, mode, theme)
    
    if "black_white_ratio" in kwargs:
        black_white_ratio = kwargs['black_white_ratio']
        black_total_ratio = black_white_ratio / (black_white_ratio + 1.0) # Calculate black-to-total ratio
        white_total_ratio = 1.0 - black_total_ratio # Calculate white-to-total ratio
        speckle_area = (np.pi * feature_size_width * feature_size_height) / 4
        total_area = screen_size_width * screen_size_height
    
        if background_colour == min(background_colour, foreground_colour): # Black background, white speckles
            total_speckles = int((white_total_ratio * total_area) / speckle_area)
        elif foreground_colour == min(background_colour, foreground_colour): # White background, black speckles
            total_speckles = int((black_total_ratio * total_area) / speckle_area)
    
        kwargs['total_speckles'] = total_speckles

    dispatch = {
        "random_disks": generate_speckles_random_disks,
        "random_disks_grid": generate_speckles_random_disks_grid,
        "perlin": generate_speckles_perlin_noise,
        "fractal": generate_speckles_perlin_noise,
        "simplex": generate_speckles_simplex_noise
    }

    generate = dispatch.get(type_gen)
    if generate is not None:
        output = generate(screen_size_width, screen_size_height, 
                    feature_size_width, feature_size_height,
                    foreground_colour, background_colour,
                    container_depth, type_gen, seed, **kwargs)
        if "black_white_ratio" in kwargs:
            image, results = output
            image = postprocess_speckles(image, seed, sigma_blur, 
                                         sigma_noise, 
                                         centre, contrast)
            return (image, results, total_speckles)
        else:
            image = output
            return output
    else:
        raise ValueError(f"Unknown speckle generation type: {type_gen}")
    
