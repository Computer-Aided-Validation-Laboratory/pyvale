import numpy as np
import argparse
import json
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
import pyvale.specklegen as specklegen

width_range = np.arange(10, 1000, 250)
height_range = np.arange(10, 1000, 250)
speckle_size_range = np.arange(4, 25, 5)
black_white_ratio_range = np.arange(0.0, 1.1, 0.2)

print(width_range.shape)
print(height_range.shape)
print(speckle_size_range.shape)
print(black_white_ratio_range.shape)

def calculate_total_speckles(bit_depth: int, width: int, height: int, 
                             theme: specklegen.Theme, black_white_ratio: float, 
                             feature_size_width: int, feature_size_height: int) -> float:

    # Calculate the total number of speckles to be generated to compare with the function output
    dynamic_range: int = 2**bit_depth - 1
    background_colour = 0 if theme == specklegen.Theme.WHITE_ON_BLACK else dynamic_range
    foreground_colour = dynamic_range if theme == specklegen.Theme.WHITE_ON_BLACK else 0
    
    black_total_ratio = black_white_ratio / (black_white_ratio + 1.0) # Calculate black-to-total ratio
    white_total_ratio = 1.0 - black_total_ratio # Calculate white-to-total ratio
    speckle_area = (np.pi * feature_size_width * feature_size_height) / 4
    total_area = width * height
    
    if background_colour == 0: # Black background, white speckles
        total_speckles_calc = int((white_total_ratio * total_area) / speckle_area)
        return total_speckles_calc
    elif foreground_colour == 0: # White background, black speckles
        total_speckles_calc = int((black_total_ratio * total_area) / speckle_area)
        return total_speckles_calc


class TestPixelsInDisk:

    sizes = list(zip(width_range, height_range, speckle_size_range))
    pytestmark = pytest.mark.parametrize("width,height,speckle_size", sizes)

    def test_pixelsInDisk_out_of_bounds(self, width: int, height: int, speckle_size: int) -> None:
        img = np.zeros((height, width), dtype=np.uint8)
        cantre_x, centre_y = -speckle_size * 2, -speckle_size * 2  # Out of bounds
        fg_colour = 1
        check_overlap = False
    
        specklegen.pixelsInDisk(cantre_x, centre_y, width, height, speckle_size, fg_colour, img, check_overlap)
    
        assert np.all(img == 0)
        
    def test_pixelsInDisk_basic_circle(self, width: int, height: int, speckle_size: int) -> None:
        img = np.zeros((height, width), dtype=np.uint8)
        cantre_x, centre_y = 5, 5
        fg_colour = 1
        check_overlap = False
    
        specklegen.pixelsInDisk(cantre_x, centre_y, width, height, speckle_size, fg_colour, img, check_overlap)
    
        # Check that the centre pixel is set
        assert img[centre_y, cantre_x] == fg_colour
    
        # Check that pixels outside the radius are not set
        radius_sq = (speckle_size / 2) ** 2
        for y in range(height):
            for x in range(width):
                dist_sq = (x + 0.5 - cantre_x) ** 2 + (y + 0.5 - centre_y) ** 2
                if dist_sq > radius_sq:
                    assert img[y, x] == 0
    
    def test_pixelsInDisk_near_edges(self, width: int, height: int, speckle_size: int) -> None:
        img = np.zeros((height, width), dtype=np.uint8)
        cantre_x, centre_y = 0, 0
        fg_colour = 1
        check_overlap = False
    
        specklegen.pixelsInDisk(cantre_x, centre_y, width, height, speckle_size, fg_colour, img, check_overlap)
    
        # Pixels outside bounds should not be accessed => Mo error => Passed test
        assert img[0, 0] == fg_colour  # Centre pixel should be set
    
        radius_sq = (speckle_size / 2) ** 2
        for y in range(height):
            for x in range(width):
                dist_sq = (x + 0.5 - cantre_x) ** 2 + (y + 0.5 - centre_y) ** 2
                if dist_sq <= radius_sq:
                    assert img[y, x] == fg_colour




class TestGenerateSpeckles:
    sizes = list(zip(width_range, height_range, speckle_size_range, black_white_ratio_range))
    pytestmark = pytest.mark.parametrize("width,height,speckle_size,black_white_ratio", sizes)
    
    def test_generate_speckles_basic_no_overlap(self, width: int, height: int, speckle_size: int, 
                                                black_white_ratio: float) -> None:
        sigma = 1.0
        bit_depth = 8
        theme = specklegen.Theme.WHITE_ON_BLACK
        feature_size_width = speckle_size
        feature_size_height = speckle_size
        seed = 123
        type_gens = ["random_disks", "random_disks_grid"]
        reduce_overlap=False
        perturbation_max = 6
    
        total_speckles_calc = calculate_total_speckles(bit_depth, width, height, 
                                                       theme, black_white_ratio, 
                                                       feature_size_width, feature_size_height)
        
        for type_gen in type_gens:
            image, results, total_speckles = specklegen.generate_speckles(width, height,
                                                   feature_size_width, feature_size_height,
                                                   theme,
                                                   bit_depth, type_gen, seed,
                                                   reduce_overlap=reduce_overlap, black_white_ratio=black_white_ratio,
                                                   sigma=sigma, perturbation_max=perturbation_max)
        
            assert total_speckles_calc == total_speckles
            assert image.shape == (height, width)
            assert results.shape == (total_speckles, 5)
        
            # All speckles should have attempt count = 1 and overlap flag = 2 (not checked)
            assert np.all(results[:, 1] == 1)
            assert np.all(results[:, 2] == 2)
    
    def test_generate_speckles_with_overlap_reduction(self, width: int, height: int, speckle_size: int, 
                                               black_white_ratio: float) -> None:
        sigma = 1.0
        bit_depth = 8
        theme = specklegen.Theme.WHITE_ON_BLACK
        feature_size_width = speckle_size
        feature_size_height = speckle_size
        seed = 123
        type_gen = "random_disks"
        reduce_overlap=True
        attempts_tot=50
    
        total_speckles_calc = calculate_total_speckles(bit_depth, width, height, 
                                                       theme, black_white_ratio, 
                                                       feature_size_width, feature_size_height)
    
        image, results, total_speckles = specklegen.generate_speckles(width, height,
                                               feature_size_width, feature_size_height,
                                               theme,
                                               bit_depth, type_gen, seed,
                                               reduce_overlap=reduce_overlap,
                                               sigma=sigma, attempts_tot=attempts_tot, black_white_ratio=black_white_ratio)
    
        assert total_speckles_calc == total_speckles
        assert image.shape == (height, width)
        assert results.shape == (total_speckles, 5)
    
        # Overlap flag should be either 0 or 1 (placed with or without overlap)
        assert np.all(np.isin(results[:, 2], [0, 1]))
    
    def test_generate_speckles_bit_depth(self, width: int, height: int, speckle_size: int, 
                                                black_white_ratio: float) -> None:
        fg_colour = 65535
        sigma = 0
        bit_depth = 16
        theme = specklegen.Theme.WHITE_ON_BLACK
        feature_size_width = speckle_size
        feature_size_height = speckle_size
        seed = 123
        type_gens = ["random_disks", "random_disks_grid"]
        reduce_overlap=False
        perturbation_max = 6
    
        total_speckles_calc = calculate_total_speckles(bit_depth, width, height, 
                                                       theme, black_white_ratio, 
                                                       feature_size_width, feature_size_height)
    
        for type_gen in type_gens:
            image, results, total_speckles = specklegen.generate_speckles(width, height,
                                                   feature_size_width, feature_size_height,
                                                   theme,
                                                   bit_depth, type_gen, seed,
                                                   reduce_overlap=reduce_overlap, black_white_ratio=black_white_ratio,
                                                   sigma=sigma, perturbation_max=perturbation_max)
        
            assert total_speckles_calc == total_speckles
            assert image.dtype == np.uint16
            assert image.max() <= fg_colour
            assert results.shape == (total_speckles, 5)
    
    def test_generate_speckles_blur(self, width: int, height: int, speckle_size: int, 
                                                black_white_ratio: float) -> None:
        fg_colour = 255
        sigma = 1.5
        bit_depth = 8
        theme = specklegen.Theme.WHITE_ON_BLACK
        feature_size_width = speckle_size
        feature_size_height = speckle_size
        seed = 123
        type_gens = ["random_disks", "random_disks_grid"]
        reduce_overlap=False
        perturbation_max = 6
    
        total_speckles_calc = calculate_total_speckles(bit_depth, width, height, 
                                                       theme, black_white_ratio, 
                                                       feature_size_width, feature_size_height)
    
        for type_gen in type_gens:
            image, results, total_speckles = specklegen.generate_speckles(width, height,
                                                   feature_size_width, feature_size_height,
                                                   theme,
                                                   bit_depth, type_gen, seed,
                                                   reduce_overlap=reduce_overlap, black_white_ratio=black_white_ratio,
                                                   sigma=sigma, perturbation_max=perturbation_max)
        
            assert total_speckles_calc == total_speckles
            assert np.any(image != 0)
            assert np.any(image != fg_colour)
            assert image.shape == (height, width)
            assert results.shape == (total_speckles, 5)

