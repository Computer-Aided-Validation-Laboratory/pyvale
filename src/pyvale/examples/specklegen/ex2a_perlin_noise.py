# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Specklegen: Speckle pattern generation using isotropic Perlin noise
================================================================================
Script to generate a synthetic speckle pattern made using isotropic Perlin noise, run diagnostics on the generated image, and save both the image
and diagnostics to the selected folder. Isotroic Perlin noise in this case means that the speckle size is the same in both horisontal and vertical directions.

This is a gradient-based noise algorithm that generates smooth and continuous random patterns. 
It produces a texture with gradually occurring transitions. 
Perlin noise achieves this by assigning random gradient vectors to grid points and then smoothly interpolating between them to create natural-looking transitions. 
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
output_path = Path.cwd() / "pyvale-output" / "ex2a"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

#%%
# Here we set the speckle pattern parameters.
# For ease of use in this example script we set parameter values directly in the
# code rather than via bash script.
# The parameters are set to generate isotropic Perlin noise in this example. 
# Perlin noise is defined by the number of noise periods width- and height-wise. 
# They can be calculated from the corresponding width and height of a feature, which is a speckle size in our case, together with a screen size.
# The noise period is obtained by dividing a screen size by a speckle size. 
# It should be noted that the screen size should be a multiple of the noise period number, otherwise the function would produce an error. 

speckle_size = 20
screen_size_width = 1000
screen_size_height = 800

image_depth = 8
container_depth = 8
mode = "scaled"

theme = specklegen.Theme.WHITE_ON_BLACK
seed = 10
type_gen = "perlin"

feature_size_width = speckle_size
feature_size_height = speckle_size

subfolder = Path(
    f"{type_gen}_{speckle_size}_{screen_size_width}_{screen_size_height}_"
    f"{image_depth}_{theme.value}_{seed}"
)
save_path = output_path / subfolder
if not os.path.exists(save_path):
    os.makedirs(save_path)

for item in save_path.iterdir():
    item.unlink()

#%%
# We now generate the speckle pattern using the specified parameters.
# The background and foreground colours are set based on the chosen theme and bit depth.
# It should be noted that there is no need to calculate the total number of speckles to generate like we did in the previous examples. 

time_start = time.time()
image = specklegen.generate_speckles(screen_size_width, screen_size_height,
                                     feature_size_width, feature_size_height,
                                     theme,
                                     image_depth, container_depth, mode, 
                                     type_gen, seed)
time_end = time.time()
time_taken = time_end - time_start

print(80*"-")
print(f"Time taken for speckle generation: {np.round(time_taken, 3)} seconds")
print(f"Subfolder: {subfolder}")
print()
    
#%%
# Now we run diagnostics on the generated speckle pattern and save the results. 
# Finally, we print out the key statistics to the console. 
# The plots are saved in the provided output folder. However, the diagnostic function outputs the matplotlib figures and axes,  
# so the plot formatting could be changed from the default one used by the function.
# The black-to-white ratio is already close to unity, 
# so there is no need to perform any additional operations related to speckle overlap reduction, like we did in ex1a, ex1b, and ex1c. 

results = specklegen.speckle_pattern_statistics(image, 
                                                image_depth, container_depth, 
                                                mode, theme)
with open(f"{save_path}/speckle_pattern_diagnostics.json", "w") as f:
    json.dump(asdict(results), f, indent=4)

image_format='jpg'
plot_opts = PlotOptsGeneral(cmap_seq='gray')
(fig,ax) = specklegen.speckle_pattern_plot(image, image_depth, container_depth, 
                                           mode, theme, plot_opts=plot_opts)
fig.savefig(f"{save_path}/speckle_pattern." + f'{image_format}', dpi=300, format=image_format, bbox_inches='tight')

# %%
# .. image:: ../../../../_static/specklegen_ex2a_speckle_pattern.jpg
#    :alt: Speckle pattern.
#    :width: 800px
#    :align: center

speckle_opts = SpecklePatternOpts(x_label="Frequency [1/pixel]",
                                  y_label="Frequency [1/pixel]",
                                  title="Spatial frequency (log scale)",
                                  cmap_title=None)
(fig,ax) = specklegen.frequency_spectrum_plot(image, 
                                              plot_opts=plot_opts,
                                              speckle_opts=speckle_opts)
fig.savefig(f"{save_path}/frequency_spectrum." + f'{image_format}', dpi=300, format=image_format, bbox_inches='tight')

# %%
# .. image:: ../../../../_static/specklegen_ex2a_frequency_spectrum.jpg
#    :alt: Frequency spectrum for the speckle pattern.
#    :width: 800px
#    :align: center

speckle_opts = SpecklePatternOpts(x_label="Pixel value",
                                  y_label="Density (log scale)",
                                  title="Histogram of irradiance values",
                                  cmap_title=None)
(fig,ax) = specklegen.pixel_value_histogram_plot(image,
                                                 speckle_opts=speckle_opts)
fig.savefig(f"{save_path}/pixel_value_histogram." + f'{image_format}', dpi=300, format=image_format, bbox_inches='tight')

# %%
# .. image:: ../../../../_static/specklegen_ex2a_pixel_value_histogram.jpg
#    :alt: Pixel value histogram for the speckle pattern.
#    :width: 800px
#    :align: center

plot_opts = PlotOptsGeneral(aspect_ratio=0.7)
speckle_opts = SpecklePatternOpts(x_label='Lag [pixels]',
                                  y_label=r"Autocov. [pixel$^2$]",
                                  title="Autocovariance",
                                  cmap_title=None)
(fig,axes) = specklegen.autocovariance_plot(image,
                                            plot_opts=plot_opts,
                                            speckle_opts=speckle_opts)
fig.savefig(f"{save_path}/autocovariance." + f'{image_format}', dpi=300, format=image_format, bbox_inches='tight')

# %%
# .. image:: ../../../../_static/specklegen_ex2a_autocovariance.jpg
#    :alt: Autocovariance for the speckle pattern.
#    :width: 800px
#    :align: center

# Uncomment this to display the plots
# plt.show()

#%%
# The relative errors beetween the specified speckle size and the speckle size approximated 
# using autocovariance are calculated. 
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
