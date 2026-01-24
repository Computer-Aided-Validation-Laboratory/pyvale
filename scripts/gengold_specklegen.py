#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import pyvale.verif.specklegold as specklegold
import pyvale.verif.specklegenconst as specklegenconst

def main() -> None:

    tags = ["random_disks", "random_disks_grid", "perlin", "fractal", "simplex"]
    
    for tag in tags:

        param_dict = {
            "speckle_size": 20,
            "screen_size_width": 1000,
            "screen_size_height": 800,
            "bit_depth": 8,
            "theme": 'white_on_black',
            "seed": 123,
            "type_gen": tag,
            "octaves": 3,
            "lacunarity": 2,
            "sigma": 4.0,
            "reduce_overlap": True,
            "attempts_tot": 300,
            "perturbation_max": 12,
            "case_tot": 3
        }
    
        print(80*"=")
        print(f"Gold Output Generator for pyvale {tag} speckle pattern generation")
        print(80*"=")
        print(f"Saving gold output to: {specklegenconst.GOLD_PATH}\n")
    
        print(f"Generating gold output for {tag} field point sensors...")
        specklegold.gen_gold_measurements(param_dict)

if __name__ == "__main__":
    main()
