# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Specklegen: Speckle pattern generation using random disk-shaped speckle placement perturbation from a grid of regularly-placed disk-shaped speckles
================================================================================
Script to generate a synthetic speckle pattern made from by randomly perturbating a grid of regularly-placed
disk-shaped speckles based on disrete uniform probability distribution, run diagnostics on the generated image, 
and save both the image and diagnostics to the selected folder.
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
output_path = Path.cwd() / "pyvale-output" / "ex1c"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

#%%
# Here we set the speckle pattern parameters.
# We aim for for approximately 50/50 black-to-white ratio.
# For ease of use in this example script we set parameter values directly in the
# code rather than via bash script.
# The parameter responsible for reducing overlap is set to 'False' in this example.
# Here we select a different type of speckle generation compared with the previous examples (ex1a and ex1b). 
# Additionally, we have an extra parameter specifying the maximum amount to move speckles by during grid perturbation (in pixels). 

speckle_size = 20
screen_size_width = 1000
screen_size_height = 800
bit_depth = 8
theme = specklegen.Theme.WHITE_ON_BLACK
black_white_ratio = 1.0
seed = 10
sigma = 4.0
reduce_overlap = False
perturbation_max = 12
type_gen = "random_disks_grid"

feature_size_width = speckle_size
feature_size_height = speckle_size

print('Start')

if reduce_overlap:
    print("Reducing overlap between speckles")
else:
    print("Not reducing overlap between speckles")

subfolder = Path(f"{type_gen}_{speckle_size}_{screen_size_width}_{screen_size_height}_{bit_depth}_{theme.value}_{seed}")
print(subfolder)
save_path = output_path / subfolder
if not os.path.exists(save_path):
    os.makedirs(save_path)

#%%
# We now generate the speckle pattern using the specified parameters.
# The background and foreground colours are set based on the chosen theme and bit depth.
# We simply pass on one additional parameter to the function. 
    
time_start = time.time()
image, results, total_speckles = specklegen.generate_speckles(screen_size_width, screen_size_height,
                                   feature_size_width, feature_size_height,
                                   theme,
                                   bit_depth, type_gen, seed,
                                   reduce_overlap=reduce_overlap,
                                   sigma=sigma, black_white_ratio=black_white_ratio,
                                   perturbation_max=perturbation_max)
time_end = time.time()
time_taken = time_end - time_start
print(f"Time taken for speckle generation: {np.round(time_taken, 3)} seconds")
print(f"Total number of speckles generated = {total_speckles}")

# save the speckle placement results
np.savetxt(f"{save_path}/speckle_placement_results.csv", results, delimiter=",", 
           header="speckle_number, attempts, overlap(1/0/2), cent_x, cent_y", comments='', fmt=['%d', '%d', '%d', '%.3f', '%.3f'])
    
#%%
# Now we run diagnostics on the generated speckle pattern and save the results. 
# Finally, we print out the key statistics to the console. 
# The plots are saved in the provided output folder. However, the diagnostic function outputs the matplotlib figures and axes,  
# so the plot formatting could be changed from the default one used by the function. 
# The black-to-white ratio is better than achieved in ex1a, when we don’t check for speckle overlap. 
# However, it is worse than the value achieved in ex1b, when we do check for speckle overlap. 
# On the other hand, we still get the benefit of the improved black-to-white ratio at the reduced computational cost, 
# as the runtime in this example is shorter than in ex1b. 

print("")
print('Starting speckle pattern diagnostics...')
results = specklegen.speckle_pattern_statistics(image, bit_depth)
plots = specklegen.speckle_pattern_plots(image, bit_depth, save_path)

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