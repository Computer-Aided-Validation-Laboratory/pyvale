"""
Specklegen: Speckle pattern generation using random disk placement with checking for overlap
================================================================================
Script to generate a synthetic speckle pattern made from randomly placed circular
speckles (disks), run diagnostics on the generated image, and save both the image
and diagnostics to the selected folder.
"""

import numpy as np
import argparse
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
import pyvale.specklegen as specklegen

#%%
# Here we parse command line arguments to set the speckle pattern parameters.
# For ease of use in this example script we set parameter values directly in the
# code rather than via bash script.
# The parameter responsible for reducing overlap is set to 'True' in this example.
parser = argparse.ArgumentParser(description='Generate random speckle patterns with specified parameters.')
args = parser.parse_args()
args.speckle_size = 25
args.screen_size_width = 1000
args.screen_size_height = 800
args.bit_depth = 8
args.theme = 'white_on_black'
args.random_seed = 10
args.sigma = 4.0
args.reduce_overlap = 'True'
args.output_path = "src/pyvale/examples/specklegen/output/ex1b"
args.attempts = None

print('Args in simulation:')
print(args)
print('')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('')

print('Start')

# Extract parameteres and revert to default values if not provided by user
speckle_size = args.speckle_size if args.speckle_size is not None else 5.0
screen_size_width = args.screen_size_width if args.screen_size_width is not None else 500
screen_size_height = args.screen_size_height if args.screen_size_height is not None else 400
bit_depth = args.bit_depth if args.bit_depth is not None else 8
theme = args.theme if args.theme is not None else 'white_on_black'
random_seed = args.random_seed if args.random_seed is not None else 10
sigma = args.sigma if args.sigma is not None else 1.0
reduce_overlap = args.reduce_overlap
attempts = args.attempts if args.attempts is not None else 100

assert bit_depth in [8, 16], "Bit depth should be either 8 or 16."
assert theme in ['black_on_white', 'white_on_black'], "Theme should be either 'black_on_white' or 'white_on_black'."
assert reduce_overlap in ['True', 'False'], "reduce_overlap should be either 'True' or 'False' (a string)."

reduce_overlap_bool = True if reduce_overlap == "True" else False
if reduce_overlap_bool:
    print("Reducing overlap between speckles")
else:
    print("Not reducing overlap between speckles")

np.random.seed(random_seed)

subfolder = f"/{speckle_size}_{screen_size_width}_{screen_size_height}_{bit_depth}_{theme}_{random_seed}_{sigma}_{reduce_overlap}"
print(subfolder)
save_path = args.output_path + subfolder
if not os.path.exists(save_path):
    os.makedirs(save_path)

#%%
# We calculate parameteres aiming for approximately 50/50 black-to-white ratio.
# We now generate the speckle pattern using the specified parameters.
# The background and foreground colours are set based on the chosen theme and bit depth.

speckle_area = np.pi * (speckle_size / 2) ** 2
total_area = screen_size_width * screen_size_height
total_speckles = int((0.5 * total_area) / speckle_area)
print(f"Total number of speckles = {total_speckles}")
dynamic_range: int = 2**bit_depth - 1
background_colour = 0 if theme == 'white_on_black' else dynamic_range
foreground_colour = dynamic_range if theme == 'white_on_black' else 0
    
# Generate speckle pattern
image, results = specklegen.generate_speckles(screen_size_width, screen_size_height, 
                 speckle_size, foreground_colour,
                 total_speckles, reduce_overlap_bool,
                 bit_depth, background_colour,
                 sigma, attempts)

# save the speckle placement results
np.savetxt(f"{save_path}/speckle_placement_results.csv", results, delimiter=",", 
           header="speckle_number, attempts, overlap(1/0/2), cent_x, cent_y", comments='', fmt=['%d', '%d', '%d', '%.3f', '%.3f'])
    
#%%
# Now we run diagnostics on the generated speckle pattern and save the results.
# Finally we print out the key statistics to the console.
print("")
print('Starting speckle pattern diagnostics...')
results = specklegen.speckle_pattern_diagnostics(image, dynamic_range, save_path)

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

error = np.abs(avg_speckle_size_fwhm - speckle_size) * 100 / speckle_size
print(f"Percentage error between requested speckle size and measured speckle size from FWHM: {np.round(error, 3)} %")
error = np.abs(avg_speckle_size_e2 - speckle_size) * 100 / speckle_size
print(f"Percentage error between requested speckle size and measured speckle size from 1/e^2: {np.round(error, 3)} %")
np.save(f"{save_path}/image.npy", image)
print("")
print('End :)')
print("")
print("")
print("")