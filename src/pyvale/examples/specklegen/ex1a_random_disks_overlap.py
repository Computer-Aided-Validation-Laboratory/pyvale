# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Specklegen: Speckle pattern generation using random disk placement without checking for overlap
================================================================================
Script to generate a synthetic speckle pattern made from randomly placed circular
speckles (disks), run diagnostics on the generated image, and save both the image
and diagnostics to the selected folder.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from dataclasses import asdict
import os
import pyvale.specklegen as specklegen
from pyvale.sensorsim.visualopts import (PlotOptsGeneral, SpecklePatternOpts)

#%%
# Let's create our standard pyvale output directory in our current working
# directory so we know where to find the files we are going to
# create.
output_path = Path.cwd() / "pyvale-output" / "ex1a"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

#%%
# Here we set the speckle pattern parameters.
# We aim for for approximately 50/50 black-to-white ratio.
# For ease of use in this example script we set parameter values directly in the
# code rather than via bash script.
# The parameter responsible for reducing overlap is set to 'False' in this example.

speckle_size = 20
screen_size_width = 1000
screen_size_height = 800
bit_depth = 8
theme = specklegen.Theme.WHITE_ON_BLACK
black_white_ratio = 1.0
seed = 10
sigma = 4.0
reduce_overlap = False
type_gen = "random_disks"

feature_size_width = speckle_size
feature_size_height = speckle_size

subfolder = Path(f"{type_gen}_{speckle_size}_{screen_size_width}_{screen_size_height}_{bit_depth}_{theme.value}_{seed}_{sigma}_{reduce_overlap}")
save_path = output_path / subfolder
if not os.path.exists(save_path):
    os.makedirs(save_path)

for item in save_path.iterdir():
    item.unlink()

#%%
# We now generate the speckle pattern using the specified parameters.
# The background and foreground colours are set based on the chosen theme and bit depth.

time_start = time.time()
image, results, total_speckles = specklegen.generate_speckles(screen_size_width, screen_size_height,
                                   feature_size_width, feature_size_height,
                                   theme,
                                   bit_depth, type_gen, seed,
                                   reduce_overlap=reduce_overlap,
                                   sigma=sigma, black_white_ratio=black_white_ratio)
time_end = time.time()
time_taken = time_end - time_start

np.savetxt(
    f"{save_path}/speckle_placement_results.csv", 
    results, 
    delimiter=",", 
    header="speckle_number, attempts, overlap(1/0/2), cent_x, cent_y", 
    comments='', 
    fmt=['%d', '%d', '%d', '%.3f', '%.3f']
)

print(80*"-")
print(f"Time taken for speckle generation: {np.round(time_taken, 3)} seconds")
print(f"Total number of speckles generated: {total_speckles}")

if reduce_overlap:
    print("Reducing overlap between speckles.")
else:
    print("Not reducing overlap between speckles.")

print(f"Subfolder: {subfolder}")
print()
    
#%%
# Now we run diagnostics on the generated speckle pattern and save the results. 
# Finally, we print out the key statistics to the console. 
# The plots are saved in the provided output folder. However, the diagnostic function outputs the matplotlib figures and axes,  
# so the plot formatting could be changed from the default one used by the function. 
# We aim to achieve black-to-white ratio as close to unity as possible. Unity ratio means 50/50 distribution of black and white colours. 
# However, in this example, black-to-white ratio considerably deviates from unity. 
# It is also visible in the irradiance value histogram. 
# The proportion of 0 irradiance values, corresponding to black colour, overweighs the 255 values, corresponding to white colour. 
# This is a result of speckle overlap.

results = specklegen.speckle_pattern_statistics(image, bit_depth)
with open(f"{save_path}/speckle_pattern_diagnostics.json", "w") as f:
    json.dump(asdict(results), f, indent=4)

image_format='jpg'
plot_opts = PlotOptsGeneral(cmap_seq='gray')
(fig,ax) = specklegen.speckle_pattern_plot(image, bit_depth, plot_opts=plot_opts)
fig.savefig(f"{save_path}/speckle_pattern." + f'{image_format}', dpi=300, format=image_format, bbox_inches='tight')

speckle_opts = SpecklePatternOpts(x_label="Frequency [1/pixel]",
                                  y_label="Frequency [1/pixel]",
                                  title="Spatial frequency (log scale)",
                                  cmap_title=None)
(fig,ax) = specklegen.frequency_spectrum_plot(image, 
                                              plot_opts=plot_opts,
                                              speckle_opts=speckle_opts)
fig.savefig(f"{save_path}/frequency_spectrum." + f'{image_format}', dpi=300, format=image_format, bbox_inches='tight')

speckle_opts = SpecklePatternOpts(x_label="Pixel value",
                                  y_label="Density (log scale)",
                                  title="Histogram of irradiance values",
                                  cmap_title=None)
(fig,ax) = specklegen.pixel_value_histogram_plot(image, bit_depth,
                                                 speckle_opts=speckle_opts)
fig.savefig(f"{save_path}/pixel_value_histogram." + f'{image_format}', dpi=300, format=image_format, bbox_inches='tight')

plot_opts = PlotOptsGeneral(aspect_ratio=0.7)
speckle_opts = SpecklePatternOpts(x_label='Lag [pixels]',
                                  y_label=r"Autocov. [pixel$^2$]",
                                  title="Autocovariance",
                                  cmap_title=None)
(fig,axes) = specklegen.autocovariance_plot(image,
                                            plot_opts=plot_opts,
                                            speckle_opts=speckle_opts)
fig.savefig(f"{save_path}/autocovariance." + f'{image_format}', dpi=300, format=image_format, bbox_inches='tight')

# Uncomment this to display the plots
# plt.show()

#%%
# The relative errors beetween the specified speckle size and the speckle size approximated using autocovariance are calculated. 
avg_speckle_size_fwhm = results.avg_speckle_size_fwhm
avg_speckle_size_e2 = results.avg_speckle_size_e2
error = np.abs(avg_speckle_size_fwhm - speckle_size) * 100 / speckle_size
error = np.abs(avg_speckle_size_e2 - speckle_size) * 100 / speckle_size
np.save(f"{save_path}/image.npy", image)
plt.imsave(f'{save_path}/image.tiff', image, cmap='gray')
plt.imsave(f'{save_path}/image.bmp', image, cmap='gray')

results_json = json.dumps(asdict(results), indent=4)

#%%
# Finally, we print the speckle statistics.
print(80*"-")
print("Speckle statistics:")
print(results_json)
print(f"Percentage error between requested speckle size and measured speckle size from FWHM: {np.round(error, 3)} %")
print(f"Percentage error between requested speckle size and measured speckle size from 1/e^2: {np.round(error, 3)} %")
print("\n"+80*"-")
