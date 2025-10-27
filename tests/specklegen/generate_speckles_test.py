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
total_speckles_range = np.arange(1, 10, 1)
sizes = list(zip(width_range, height_range, speckle_size_range))

print(width_range.shape)
print(height_range.shape)
print(speckle_size_range.shape)

pytestmark = pytest.mark.parametrize("width,height,speckle_size", sizes)

def test_pixelsInDisk_out_of_bounds(width: int, height: int, speckle_size: int) -> None:
    img = np.zeros((height, width), dtype=np.uint8)
    cantre_x, centre_y = -speckle_size * 2, -speckle_size * 2  # Out of bounds
    fg_colour = 1
    check_overlap = False

    specklegen.pixelsInDisk(cantre_x, centre_y, width, height, speckle_size, fg_colour, img, check_overlap)

    assert np.all(img == 0)
    
def test_pixelsInDisk_basic_circle(width: int, height: int, speckle_size: int) -> None:
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

def test_pixelsInDisk_near_edges(width: int, height: int, speckle_size: int) -> None:
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

@pytest.mark.parametrize("total_speckles", total_speckles_range)
def test_generate_speckles_basic_no_overlap(width: int, height: int, speckle_size: int, 
                                            total_speckles: int) -> None:
    fg_colour = 255
    bg_colour = 0
    total_speckles = 5
    sigma = 1.0
    bit_depth = 8

    image, results = specklegen.generate_speckles(width, height, speckle_size, fg_colour, total_speckles, 
                                      reduce_overlap=False, bit_depth=bit_depth,
                                      background_colour=bg_colour, sigma=sigma)

    assert image.shape == (height, width)
    assert results.shape == (total_speckles, 5)

    # All speckles should have attempt count = 1 and overlap flag = 2 (not checked)
    assert np.all(results[:, 1] == 1)
    assert np.all(results[:, 2] == 2)


@pytest.mark.parametrize("total_speckles", total_speckles_range)
def test_generate_speckles_with_overlap_reduction(width: int, height: int, speckle_size: int, 
                                            total_speckles: int) -> None:
    fg_colour = 255
    bg_colour = 0
    sigma = 1.0
    bit_depth = 8

    image, results = specklegen.generate_speckles(width, height, speckle_size, fg_colour, total_speckles, 
                                      reduce_overlap=True, bit_depth=bit_depth,
                                      background_colour=bg_colour, sigma=sigma, attempts_tot=50)

    assert image.shape == (height, width)
    assert results.shape == (total_speckles, 5)

    # Overlap flag should be either 0 or 1 (placed with or without overlap)
    assert np.all(np.isin(results[:, 2], [0, 1]))

@pytest.mark.parametrize("total_speckles", total_speckles_range)
def test_generate_speckles_bit_depth(width: int, height: int, speckle_size: int, 
                                            total_speckles: int) -> None:
    fg_colour = 65535
    bg_colour = 0
    sigma = 0
    bit_depth = 16

    image, results = specklegen.generate_speckles(width, height, speckle_size, fg_colour, total_speckles,
                                      reduce_overlap=False, bit_depth=bit_depth,
                                      background_colour=bg_colour, sigma=sigma)

    assert image.dtype == np.uint16
    assert image.max() <= fg_colour
    assert results.shape == (total_speckles, 5)

@pytest.mark.parametrize("total_speckles", total_speckles_range)
def test_generate_speckles_blur(width: int, height: int, speckle_size: int, 
                                            total_speckles: int) -> None:
    fg_colour = 255
    bg_colour = 0
    sigma = 1.5
    bit_depth = 8

    image, results = specklegen.generate_speckles(width, height, speckle_size, fg_colour, total_speckles,
                                      reduce_overlap=False, bit_depth=bit_depth,
                                      background_colour=bg_colour, sigma=sigma)

    assert np.any(image != 0)
    assert np.any(image != fg_colour)
    assert image.shape == (height, width)
    assert results.shape == (total_speckles, 5)
