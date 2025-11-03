#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

"""
DEVELOPER VERIFICATION MODULE
--------------------------------------------------------------------------------
This module contains developer utility functions used for verification testing
of the speckle generation pattern in pyvale.

Specifically, this module contains gold measurement generation function.
"""

import numpy as np
# import pyvale.verif.specklegneconst as specklegneconst


import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import src.pyvale.verif.specklegneconst as specklegneconst
import src.pyvale.specklegen as specklegen


def gen_gold_measurements(param_dict: dict) -> None:
    case_tot = param_dict.get("case_tot")
    type_gen = param_dict.get("type_gen")

    screen_size_width = param_dict.get("screen_size_width")
    screen_size_height = param_dict.get("screen_size_height")
    speckle_size = param_dict.get("speckle_size")
    bit_depth = param_dict.get("bit_depth")
    theme = param_dict.get("theme")
    seed = param_dict.get("seed")
    reduce_overlap = param_dict.get("reduce_overlap")
    sigma = param_dict.get("sigma")
    attempts_tot = param_dict.get("attempts_tot")
    perturbation_max = param_dict.get("perturbation_max")
    octaves = param_dict.get("octaves")
    lacunarity = param_dict.get("lacunarity")
    feature_size_width = speckle_size
    feature_size_height = speckle_size

    speckle_area = np.pi * (speckle_size / 2) ** 2
    total_area = screen_size_width * screen_size_height
    total_speckles = int((0.5 * total_area) / speckle_area)
    dynamic_range: int = 2**bit_depth - 1
    background_colour = 0 if theme == 'white_on_black' else dynamic_range
    foreground_colour = dynamic_range if theme == 'white_on_black' else 0


    for i in range(case_tot):
        print(f"Generating gold output for case: {type_gen}_{i+1}")
 
        if type_gen == "random_disks":
            image, _ = specklegen.generate_speckles(screen_size_width, screen_size_height,
                                            feature_size_width, feature_size_height,
                                            foreground_colour, background_colour,
                                            bit_depth, type_gen, seed,
                                            total_speckles=total_speckles,
                                            reduce_overlap=reduce_overlap,
                                            sigma=sigma, attempts_tot=attempts_tot,
                                            perturbation_max=perturbation_max)
        elif type_gen == "random_disks_grid":
            image, _ = specklegen.generate_speckles(screen_size_width, screen_size_height,
                                        feature_size_width, feature_size_height,
                                        foreground_colour, background_colour,
                                        bit_depth, type_gen, seed,
                                        total_speckles=total_speckles,
                                        reduce_overlap=reduce_overlap,
                                        sigma=sigma, perturbation_max=perturbation_max)
        elif type_gen == "perlin":
            image = specklegen.generate_speckles(screen_size_width, screen_size_height,
                                        feature_size_width, feature_size_height,
                                        foreground_colour, background_colour,
                                        bit_depth, type_gen, seed)
        elif type_gen == "fractal":
            image = specklegen.generate_speckles(screen_size_width, screen_size_height,
                                        feature_size_width, feature_size_height,
                                        foreground_colour, background_colour,
                                        bit_depth, type_gen, seed,
                                        octaves=octaves, lacunarity=lacunarity)
            
        elif type_gen == "simplex":
            image = specklegen.generate_speckles(screen_size_width, screen_size_height,
                                         feature_size_width, feature_size_height,
                                         foreground_colour, background_colour,
                                         bit_depth, type_gen, seed)
 
        save_path = specklegneconst.GOLD_PATH
        if not os.path.exists(save_path):
           os.makedirs(save_path)
        np.save(f"{save_path}/{type_gen}_image_{i+1}.npy", image)

        if i == 0:
            image_ref = image
        else:
            error_abs = np.abs(image - image_ref)
            np.save(f"{save_path}/{type_gen}_image_abs_error_{i+1}.npy", image)
            print(f"Avg. absolute error {np.mean(error_abs)}, max. absolute error {np.max(error_abs)}")


