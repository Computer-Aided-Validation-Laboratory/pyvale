import numpy as np
import argparse
import json
import pytest
import itertools
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
import pyvale.specklegen as specklegen

# @pytest.fixture(params=list(zip(
#     np.arange(10, 1000, 250), # width
#     np.arange(10, 1000, 250), # height
#     np.arange(4, 25, 5),      # speckle_size
#     np.arange(0.0, 1.1, 0.2),  # black_white_ratio
#     np.array((8, 10, 12, 16, 32)), # image depth
#     )))

# @pytest.fixture(params=list(itertools.product(
#     np.array((800, 1000)), # width
#     np.array((600, 800)), # height
#     np.array((10, 25)),      # speckle_size
#     np.array((0.4, 0.6)),  # black_white_ratio
#     np.array((8, 10, 12, 16, 32)), # image depth
#     )))

# @pytest.fixture(params=list(zip(
#     np.array((800, 800, 800, 800, 800)), # width
#     np.array((600, 600, 600, 600, 600)), # height
#     np.array((25, 25, 25, 25, 25)),      # speckle_size
#     np.array((1.0, 1.0, 1.0, 1.0, 1.0)),  # black_white_ratio
#     np.array((8, 10, 12, 16, 32)), # image depth
#     np.array((8, 16, 32, 32, 32)), # container depth
#     np.array(("scaled", "scaled", "scaled", "scaled", "scaled")) # mode
#     )))


@pytest.fixture(params=list(itertools.product(
    np.array((800, )), # width
    np.array((600, )), # height
    np.array((25, )),      # speckle_size
    np.array((1.0, )),  # black_white_ratio
    np.array((8, 10, 12, 16, 32)), # image depth
    np.array((8, 16, 32)), # container depth
    np.array(("scaled", "upper", "lower")), # mode
    np.array((1.0, 1.5)) # sigma blur
    )))
def image_dims(request):
    # width, height, speckle_size, black_white_ratio
    return request.param

def calculate_total_speckles(image_depth: int, container_depth: int, mode: str, 
                             width: int, height: int, 
                             theme: specklegen.Theme, black_white_ratio: float, 
                             feature_size_width: int, feature_size_height: int) -> float:

    # Calculate the total number of speckles to be generated to compare with the function output

    background_colour, foreground_colour = specklegen.get_colours(image_depth, container_depth, 
                                                                  mode, theme)
    
    # Calculate black-to-total ratio
    black_total_ratio = black_white_ratio / (black_white_ratio + 1.0)
    # Calculate white-to-total ratio
    white_total_ratio = 1.0 - black_total_ratio
    speckle_area = (np.pi * feature_size_width * feature_size_height) / 4
    total_area = width * height
    
    if background_colour == min(background_colour, foreground_colour): 
        # Black background, white speckles
        total_speckles_calc = int((white_total_ratio * total_area) / speckle_area)
        return total_speckles_calc
    elif foreground_colour == min(background_colour, foreground_colour): 
        # White background, black speckles
        total_speckles_calc = int((black_total_ratio * total_area) / speckle_area)
        return total_speckles_calc

class TestGenerateSpeckles:
    
    def test_generate_speckles_basic_no_overlap(self, image_dims: tuple) -> None:
        width, height, speckle_size, black_white_ratio, \
            image_depth, container_depth, mode, sigma_blur = image_dims

        theme = specklegen.Theme.WHITE_ON_BLACK
        feature_size_width = speckle_size
        feature_size_height = speckle_size
        seed = 123
        type_gens = ["random_disks", "random_disks_grid"]
        reduce_overlap=False
        perturbation_max = 6
        
        for type_gen in type_gens:
            if image_depth <= container_depth:
                total_speckles_calc = calculate_total_speckles(image_depth, 
                                                               container_depth, mode,
                                                               width, height,
                                                               theme, black_white_ratio, 
                                                               feature_size_width, feature_size_height)
                background_colour, foreground_colour = specklegen.get_colours(image_depth, 
                                                                              container_depth, 
                                                                              mode, theme)
                image, results, total_speckles = \
                    specklegen.generate_speckles(width, height,
                                                 feature_size_width, feature_size_height,
                                                 theme,
                                                 image_depth, container_depth, mode,
                                                 type_gen, seed,
                                                 reduce_overlap=reduce_overlap, 
                                                 black_white_ratio=black_white_ratio,
                                                 sigma_blur=sigma_blur, 
                                                 perturbation_max=perturbation_max)
                
                assert total_speckles_calc == total_speckles
                assert image.shape == (height, width)
                assert image.dtype == specklegen.DTYPE_MAP.get(container_depth)
                assert np.min(image) >= min(background_colour, foreground_colour)
                assert np.max(image) <= max(background_colour, foreground_colour)
                assert results.shape == (total_speckles, 5)
            
                # All speckles should have attempt count = 1 and overlap flag = 2 (not checked)
                assert np.all(results[:, 1] == 1)
                assert np.all(results[:, 2] == 2)
            else:
                with pytest.raises(AssertionError):
                    image, results, total_speckles = \
                        specklegen.generate_speckles(width, height,
                                                     feature_size_width, feature_size_height,
                                                     theme,
                                                     image_depth, container_depth, mode,
                                                     type_gen, seed,
                                                     reduce_overlap=reduce_overlap, 
                                                     black_white_ratio=black_white_ratio,
                                                     sigma_blur=sigma_blur, 
                                                     perturbation_max=perturbation_max)

            
    
    def test_generate_speckles_with_overlap_reduction(self, image_dims: tuple) -> None:
        width, height, speckle_size, black_white_ratio, \
            image_depth, container_depth, mode, sigma_blur = image_dims

        theme = specklegen.Theme.WHITE_ON_BLACK
        feature_size_width = speckle_size
        feature_size_height = speckle_size
        seed = 123
        type_gen = "random_disks"
        reduce_overlap=True
        attempts_tot=50
    
        if image_depth <= container_depth:
            total_speckles_calc = calculate_total_speckles(image_depth, 
                                                           container_depth, mode,
                                                           width, height,
                                                           theme, black_white_ratio, 
                                                           feature_size_width, feature_size_height)
            background_colour, foreground_colour = specklegen.get_colours(image_depth, 
                                                              container_depth, 
                                                              mode, theme)
        
            image, results, total_speckles = \
                  specklegen.generate_speckles(width, height,
                                               feature_size_width, feature_size_height,
                                               theme,
                                               image_depth, container_depth, mode, 
                                               type_gen, seed,
                                               reduce_overlap=reduce_overlap,
                                               black_white_ratio=black_white_ratio,
                                               sigma_blur=sigma_blur, 
                                               attempts_tot=attempts_tot)
        
            assert total_speckles_calc == total_speckles
            assert image.shape == (height, width)
            assert results.shape == (total_speckles, 5)
            assert image.dtype == specklegen.DTYPE_MAP.get(container_depth)
            assert np.min(image) >= min(background_colour, foreground_colour)
            assert np.max(image) <= max(background_colour, foreground_colour)
        
            # Overlap flag should be either 0 or 1 (placed with or without overlap)
            assert np.all(np.isin(results[:, 2], [0, 1]))
        
        else:
                with pytest.raises(AssertionError):
                                image, results, total_speckles = \
                                    specklegen.generate_speckles(width, height,
                                                                 feature_size_width, feature_size_height,
                                                                 theme,
                                                                 image_depth, container_depth, mode, 
                                                                 type_gen, seed,
                                                                 reduce_overlap=reduce_overlap,
                                                                 black_white_ratio=black_white_ratio,
                                                                 sigma_blur=sigma_blur, 
                                                                 attempts_tot=attempts_tot)
    
    
    
    def test_generate_speckles_centre_contrast(self, image_dims: tuple) -> None:
        width, height, speckle_size, black_white_ratio, \
            image_depth, container_depth, mode, sigma_blur = image_dims
        sigma_blur = 0.0
        centre = 120
        contrast = 100

        theme = specklegen.Theme.WHITE_ON_BLACK
        feature_size_width = speckle_size
        feature_size_height = speckle_size
        seed = 123
        type_gens = ["random_disks", "random_disks_grid", "perlin", "simplex"]
        reduce_overlap = False
        perturbation_max = 6

        v_max = centre + (contrast / 2)
        v_min = centre - (contrast / 2)
    
        for type_gen in type_gens:
            if image_depth <= container_depth:
                total_speckles_calc = calculate_total_speckles(image_depth, 
                                                               container_depth, mode,
                                                               width, height,
                                                               theme, black_white_ratio, 
                                                               feature_size_width, feature_size_height)
                output = \
                    specklegen.generate_speckles(width, height,
                                                 feature_size_width, feature_size_height,
                                                 theme,
                                                 image_depth, container_depth, mode, 
                                                 type_gen, seed,
                                                 reduce_overlap=reduce_overlap, 
                                                 black_white_ratio=black_white_ratio,
                                                 sigma_blur=sigma_blur,
                                                 centre=centre, contrast=contrast, 
                                                 perturbation_max=perturbation_max)
                
                if isinstance(output, (list, tuple)):
                    image, results, total_speckles = output
                else:
                    image = output
                    results = None
                    total_speckles = None
            
                if results is not None:
                     assert total_speckles_calc == total_speckles
                     assert results.shape == (total_speckles, 5)
                
                assert image.shape == (height, width)
                assert image.dtype == specklegen.DTYPE_MAP.get(container_depth)
                assert np.max(image) == v_max
                assert np.min(image) == v_min
            else:
                with pytest.raises(AssertionError):
                    specklegen.generate_speckles(width, height,
                                                 feature_size_width, feature_size_height,
                                                 theme,
                                                 image_depth, container_depth, mode, 
                                                 type_gen, seed,
                                                 reduce_overlap=reduce_overlap, 
                                                 black_white_ratio=black_white_ratio,
                                                 sigma_blur=sigma_blur,
                                                 centre=centre, contrast=contrast, 
                                                 perturbation_max=perturbation_max)
                                
    def test_generate_speckles_noise(self, image_dims: tuple) -> None:
        width, height, speckle_size, black_white_ratio, \
            image_depth, container_depth, mode, sigma_blur = image_dims
        
        sigma_blur = 0.0
        sigma_noise = 6.0

        theme = specklegen.Theme.WHITE_ON_BLACK
        feature_size_width = speckle_size
        feature_size_height = speckle_size
        seed = 123
        type_gens = ["random_disks", "random_disks_grid", "perlin", "simplex"]
        reduce_overlap = False
        perturbation_max = 6
    
        for type_gen in type_gens:
            if image_depth <= container_depth:
                total_speckles_calc = calculate_total_speckles(image_depth, 
                                                               container_depth, mode,
                                                               width, height,
                                                               theme, black_white_ratio, 
                                                               feature_size_width, feature_size_height)
                background_colour, foreground_colour = specklegen.get_colours(image_depth, 
                                                                              container_depth, 
                                                                              mode, theme)
                output = \
                    specklegen.generate_speckles(width, height,
                                                 feature_size_width, feature_size_height,
                                                 theme,
                                                 image_depth, container_depth, mode, 
                                                 type_gen, seed,
                                                 reduce_overlap=reduce_overlap, 
                                                 black_white_ratio=black_white_ratio,
                                                 sigma_blur=sigma_blur,
                                                 sigma_noise=sigma_noise, 
                                                 perturbation_max=perturbation_max)
                
                if isinstance(output, (list, tuple)):
                    image, results, total_speckles = output
                else:
                    image = output
                    results = None
                    total_speckles = None
            
                
                if results is not None:
                    assert total_speckles_calc == total_speckles
                    assert results.shape == (total_speckles, 5)
                
                assert image.shape == (height, width)
                assert image.dtype == specklegen.DTYPE_MAP.get(container_depth)
                assert np.min(image) >= min(background_colour, foreground_colour)
                assert np.max(image) <= max(background_colour, foreground_colour)
            else:
                with pytest.raises(AssertionError):
                     specklegen.generate_speckles(width, height,
                                                  feature_size_width, feature_size_height,
                                                  theme,
                                                  image_depth, container_depth, mode, 
                                                  type_gen, seed,
                                                  reduce_overlap=reduce_overlap, 
                                                  black_white_ratio=black_white_ratio,
                                                  sigma_blur=sigma_blur,
                                                  sigma_noise=sigma_noise, 
                                                  perturbation_max=perturbation_max)

