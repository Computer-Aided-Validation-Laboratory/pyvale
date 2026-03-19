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
import os
import pyvale.verif.specklegenconst as specklegenconst
import pyvale.specklegen as specklegen


def gen_gold_measurements(param_dict: dict) -> None:
    case_tot = param_dict.get("case_tot")
    type_gen = param_dict.get("type_gen")

    screen_size_width = param_dict.get("screen_size_width")
    screen_size_height = param_dict.get("screen_size_height")
    speckle_size = param_dict.get("speckle_size")
    image_depth = param_dict.get("image_depth")
    container_depth = param_dict.get("container_depth")
    mode = param_dict.get("mode")
    theme = param_dict.get("theme")
    black_white_ratio = param_dict.get("black_white_ratio")
    seed = param_dict.get("seed")
    reduce_overlap = param_dict.get("reduce_overlap")
    sigma_blur = param_dict.get("sigma_blur")
    attempts_tot = param_dict.get("attempts_tot")
    perturbation_max = param_dict.get("perturbation_max")
    octaves = param_dict.get("octaves")
    lacunarity = param_dict.get("lacunarity")
    
    feature_size_width = speckle_size
    feature_size_height = speckle_size

    for i in range(case_tot):
        print(f"Generating gold output for case: {type_gen}_{i+1}")
 
        if type_gen == "random_disks":
            image, _, _ = \
                specklegen.generate_speckles(screen_size_width, screen_size_height,
                                             feature_size_width, feature_size_height,
                                             theme,
                                             image_depth, container_depth, mode, 
                                             type_gen, seed,
                                             reduce_overlap=reduce_overlap,
                                             sigma_blur=sigma_blur, 
                                             black_white_ratio=black_white_ratio,
                                             attempts_tot=attempts_tot,
                                             perturbation_max=perturbation_max)
        
        elif type_gen == "random_disks_grid":
            image, _, _ = \
                specklegen.generate_speckles(screen_size_width, screen_size_height,
                                             feature_size_width, feature_size_height,
                                             theme,
                                             image_depth, container_depth, mode, 
                                             type_gen, seed,
                                             reduce_overlap=reduce_overlap,
                                             sigma_blur=sigma_blur, 
                                             black_white_ratio=black_white_ratio,
                                             attempts_tot=attempts_tot,
                                             perturbation_max=perturbation_max)
        elif type_gen == "perlin":
            image = specklegen.generate_speckles(screen_size_width, screen_size_height,
                                                 feature_size_width, feature_size_height,
                                                 theme,
                                                 image_depth, container_depth, mode, 
                                                 type_gen, seed)
        elif type_gen == "fractal":
            image = specklegen.generate_speckles(screen_size_width, screen_size_height,
                                                 feature_size_width, feature_size_height,
                                                 theme,
                                                 image_depth, container_depth, mode, 
                                                 type_gen, seed,
                                                 octaves=octaves, lacunarity=lacunarity)
            
        elif type_gen == "simplex": 
            image = specklegen.generate_speckles(screen_size_width, screen_size_height,
                                                 feature_size_width, feature_size_height,
                                                 theme,
                                                 image_depth, container_depth, mode, 
                                                 type_gen, seed)
 
        save_path = specklegenconst.GOLD_PATH
        if not os.path.exists(save_path):
           os.makedirs(save_path)
        np.save(f"{save_path}/{type_gen}_image_{i+1}.npy", image)

        if i == 0:
            image_ref = image
        else:
            error_abs = np.abs(image - image_ref)
            np.save(f"{save_path}/{type_gen}_image_abs_error_{i+1}.npy", image)
            print(f"Avg. absolute error {np.mean(error_abs)}, max. absolute error {np.max(error_abs)}")


