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

import numpy as np
import time
import json
import os
import pyvale.specklegen as specklegen

#%%
# Here we parse command line arguments to set the speckle pattern parameters.
# For ease of use in this example script we set parameter values directly in the
# code rather than via bash script.
# The parameters are set to generate isotropic Simplex noise in this example.
# Contrary to the Perlin (ex2a) and fractal (ex3a) noise, there are no restrictions placed on the parameters defining Simplex noise.

speckle_size = 20
screen_size_width = 1000
screen_size_height = 800
bit_depth = 8
theme = 'white_on_black'
seed = 10
type_gen = "simplex"
output_path = "src/pyvale/examples/specklegen/output/ex4a"

print('Start')

assert theme in ['black_on_white', 'white_on_black'], "Theme should be either 'black_on_white' or 'white_on_black'."

subfolder = f"/{type_gen}_{speckle_size}_{screen_size_width}_{screen_size_height}_{bit_depth}_{theme}_{seed}"
print(subfolder)
save_path = output_path + subfolder
if not os.path.exists(save_path):
    os.makedirs(save_path)

#%%
# We now generate the speckle pattern using the specified parameters.
# The background and foreground colours are set based on the chosen theme and bit depth.

dynamic_range: int = 2**bit_depth - 1
background_colour = 0 if theme == 'white_on_black' else dynamic_range
foreground_colour = dynamic_range if theme == 'white_on_black' else 0

feature_size_width = speckle_size
feature_size_height = speckle_size
    
# Generate speckle pattern
time_start = time.time()
image = specklegen.generate_speckles(screen_size_width, screen_size_height,
                                     feature_size_width, feature_size_height,
                                     foreground_colour, background_colour,
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
results = specklegen.speckle_pattern_statistics(image, dynamic_range)
plots = specklegen.speckle_pattern_plots(image, dynamic_range, save_path)

# save the diagnostics results
with open(f"{save_path}/speckle_pattern_diagnostics.json", 'w') as f:
    json.dump(results, f, indent=4)

ratio = results.get("black_white_ratio", None)
mean_gradient = results.get("mean_intensity_gradient", None)
std_dev = results.get("std_dev_irradiance", None)
avg = results.get("avg_irradiance", None)
contrast = results.get("contrast", None)
entropy = results.get("shannon_entropy", None)
peak_to_mean = results.get("peak_to_mean_ratio", None)
skew = results.get("skewness", None)
kurt = results.get("kurtosis", None)
avg_speckle_size_fwhm = results.get("avg_speckle_size_fwhm", None)
avg_speckle_size_e2 = results.get("avg_speckle_size_e2", None)
H_fit_stats = results.get("H_fit_stats", None)
V_fit_stats = results.get("V_fit_stats", None)

print("")
print("Speckle statistics:")

print(f"Black/White ratio: {np.round(ratio, 3)}")
print(f"Mean intensity gradient: {np.round(mean_gradient, 3)}")
print(f"Standard deviation of irradiance values: {np.round(std_dev, 3)}")
print(f"Average irradiance value: {np.round(avg, 3)}")
print(f"Contrast (std/mean): {np.round(contrast, 3)}")
print(f"Skewness: {np.round(skew, 3)}")
print(f"Kurtosis: {np.round(kurt, 3)}")
print(f"Shannon entropy: {np.round(entropy, 3)}")
print(f"Peak to mean ratio: {np.round(peak_to_mean, 3)}")
print(f"Average speckle size (full width at half maximum): {np.round(avg_speckle_size_fwhm, 3)} pixels")
print(f"Average speckle size (1/e^2): {np.round(avg_speckle_size_e2, 3)} pixels")
print(f"R_squared: Horisontal fit: {np.round(H_fit_stats['R_squared'], 3)}, Vertical fit: {np.round(V_fit_stats['R_squared'], 3)}")

# The relative errors beetween the specified speckle size and the speckle size approximated using cautocovariance are calculated. 
error = np.abs(avg_speckle_size_fwhm - speckle_size) * 100 / speckle_size
print(f"Percentage error between requested speckle size and measured speckle size from FWHM: {np.round(error, 3)} %")
error = np.abs(avg_speckle_size_e2 - speckle_size) * 100 / speckle_size
print(f"Percentage error between requested speckle size and measured speckle size from 1/e^2: {np.round(error, 3)} %")
np.save(f"{save_path}/image.npy", image)
print("")
print('End :)')
print("")