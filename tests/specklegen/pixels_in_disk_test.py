import numpy as np
import argparse
import json
import pytest
import itertools
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
import pyvale.specklegen as specklegen

@pytest.fixture(params=list(zip(
    np.array((10, 1000, 250)), # width
    np.array((10, 1000, 250)), # height
    np.array((4, 25, 5)),      # speckle_size
    )))
def image_dims(request):
    # width, height, speckle_size
    return request.param

class TestPixelsInDisk:

    def test_pixelsInDisk_out_of_bounds(self, image_dims: tuple) -> None:
        width, height, speckle_size = image_dims
        img = np.zeros((height, width), dtype=np.uint8)
        cantre_x, centre_y = -speckle_size * 2, -speckle_size * 2  # Out of bounds
        fg_colour = 1
        check_overlap = False
    
        specklegen.pixelsInDisk(cantre_x, centre_y, width, height, 
                                speckle_size, fg_colour, img, check_overlap)
    
        assert np.all(img == 0)
        
    def test_pixelsInDisk_basic_circle(self, image_dims: tuple) -> None:
        width, height, speckle_size = image_dims
        img = np.zeros((height, width), dtype=np.uint8)
        cantre_x, centre_y = 5, 5
        fg_colour = 1
        check_overlap = False
    
        specklegen.pixelsInDisk(cantre_x, centre_y, width, height, 
                                speckle_size, fg_colour, img, check_overlap)
    
        # Check that the centre pixel is set
        assert img[centre_y, cantre_x] == fg_colour
    
        # Check that pixels outside the radius are not set
        radius_sq = (speckle_size / 2) ** 2
        for y in range(height):
            for x in range(width):
                dist_sq = (x + 0.5 - cantre_x) ** 2 + (y + 0.5 - centre_y) ** 2
                if dist_sq > radius_sq:
                    assert img[y, x] == 0
    
    def test_pixelsInDisk_near_edges(self, image_dims: tuple) -> None:
        width, height, speckle_size = image_dims
        img = np.zeros((height, width), dtype=np.uint8)
        cantre_x, centre_y = 0, 0
        fg_colour = 1
        check_overlap = False
    
        specklegen.pixelsInDisk(cantre_x, centre_y, width, height, 
                                speckle_size, fg_colour, img, check_overlap)
    
        # Pixels outside bounds should not be accessed => Mo error => Passed test
        assert img[0, 0] == fg_colour  # Centre pixel should be set
    
        radius_sq = (speckle_size / 2) ** 2
        for y in range(height):
            for x in range(width):
                dist_sq = (x + 0.5 - cantre_x) ** 2 + (y + 0.5 - centre_y) ** 2
                if dist_sq <= radius_sq:
                    assert img[y, x] == fg_colour