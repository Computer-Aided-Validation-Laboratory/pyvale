# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Specklegen: Speckle pattern generation using isotropic Simplex noise
================================================================================
Script to generate a synthetic speckle pattern made using isotropic Simplex noise, run diagnostics on the generated image, and save both the image
and diagnostics to the selected folder. Isotroic Simplex noise in this case means that the speckle size is the same in both horisontal and vertical directions.

Simplex noise is an enhanced version of Perlin noise that aims to produce more consistent and isotropic noise patterns.
"""

from pathlib import Path
import numpy as np
import time
import json
import os
import pyvale.specklegen as specklegen

#%%
# Let's create our standard pyvale output directory in our current working
# directory so we know where to find the files we are going to
# create.
output_path = Path.cwd() / "pyvale-output" / "ex4a"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

#%%
# Here we set the speckle pattern parameters.
# For ease of use in this example script we set parameter values directly in the
# code rather than via bash script.
# The parameters are set to generate isotropic Simplex noise in this example.
# Contrary to the Perlin (ex2a) and fractal (ex3a) noise, there are no restrictions placed on the parameters defining Simplex noise.

speckle_size = 20
screen_size_width = 1000
screen_size_height = 800
bit_depth = 8
theme = specklegen.Theme.WHITE_ON_BLACK
seed = 10
type_gen = "simplex"

feature_size_width = speckle_size
feature_size_height = speckle_size

print('Start')

subfolder = Path(f"{type_gen}_{speckle_size}_{screen_size_width}_{screen_size_height}_{bit_depth}_{theme.value}_{seed}")
print(subfolder)
save_path = output_path / subfolder
if not os.path.exists(save_path):
    os.makedirs(save_path)

#%%
# We now generate the speckle pattern using the specified parameters.
# The background and foreground colours are set based on the chosen theme and bit depth.
    
time_start = time.time()
image = specklegen.generate_speckles(screen_size_width, screen_size_height,
                                     feature_size_width, feature_size_height,
                                     theme,
                                     bit_depth, type_gen, seed)
time_end = time.time()
time_taken = time_end - time_start
print(f"Time taken for speckle generation: {np.round(time_taken, 3)} seconds")
    
#%%
# Now we run diagnostics on the generated speckle pattern and save the results. 
# Finally, we print out the key statistics to the console. 
# The plots are saved in the provided output folder. However, the diagnostic function outputs the matplotlib figures and axes,  
# so the plot formatting could be changed from the default one used by the function.
# The black-to-white ratio is already close to unity, 
# so there is no need to perform any additional operations related to speckle overlap reduction, like we did in ex1a, ex1b, and ex1c. 

print("")
print('Starting speckle pattern diagnostics...')
results = specklegen.speckle_pattern_statistics(image, bit_depth)
plots = specklegen.speckle_pattern_plots(image, bit_depth, save_path)

with open(f"{save_path}/speckle_pattern_diagnostics.json", 'w') as f:
    json.dump(results, f, indent=4)

avg_speckle_size_fwhm = results.get("avg_speckle_size_fwhm", None)
avg_speckle_size_e2 = results.get("avg_speckle_size_e2", None)

print("")
print("Speckle statistics:")

for key,value in results.items():
    if isinstance(value, (float, np.floating)):
        display_value = np.round(value, 2)
    else:
        display_value = value
        
    print(f"{key}: {display_value}")

#%%
# Finally, the relative errors beetween the specified speckle size and the speckle size approximated using autocovariance are calculated. 
error = np.abs(avg_speckle_size_fwhm - speckle_size) * 100 / speckle_size
print(f"Percentage error between requested speckle size and measured speckle size from FWHM: {np.round(error, 3)} %")
error = np.abs(avg_speckle_size_e2 - speckle_size) * 100 / speckle_size
print(f"Percentage error between requested speckle size and measured speckle size from 1/e^2: {np.round(error, 3)} %")
np.save(f"{save_path}/image.npy", image)
print("")
print('End :)')
print("")