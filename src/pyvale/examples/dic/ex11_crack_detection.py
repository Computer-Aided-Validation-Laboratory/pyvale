#================================================================================
#Example: thermocouples on a 2d plate
#
#pyvale: the python validation engine
#License: MIT
#Copyright (C) 2024 The Computer Aided Validation Team
#================================================================================
"""
Crack Detection using Gray Level Residuals
---------------------

In this example we'll look at identifying cracks in a 2D plate using the gray
level residuals from a DIC analysis. The gray level residuals are calculated as
the difference between the reference and deformed images after applying the DIC
displacement field. Areas with high residuals can indicate potential cracks or
defects in the material. The methods used in pyvale for calculating the gray
level residual is based on the work of `Chi et al.
(2026) <https://iopscience.iop.org/article/10.1088/1361-6501/accbdf>`_. 

For this example we'll use an open source pair of images from the `Fatigue
Monitoring of Composites (F-MOC) dataset
<https://data.mendeley.com/datasets/4zm6jh8jkd/1>`_ created by Panagiotis Komninos, 
Athanasios Kontogiannis , Nick Eleftheroglou and Dimitrios Zarouchas (2024):
"""

# %%
# .. list-table::
#    :widths: 50 50
#    :align: center
#
#    * - **Reference**
#      - **Deformed**
#    * - .. image:: ../../../../_static/dic_ex11_ref.png
#           :width: 100%
#      - .. image:: ../../../../_static/dic_ex11_def.png
#           :width: 100%

# pyvale modules
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from pyvale import dic
from pyvale import data

# %%
# load the ROI for this example:

# image are not present in the deformed image.
roi = dic.RegionOfInterest(data.dic_ex11_ref())
roi.read_yaml(data.dic_ex11_roi())

# create a directory for the the different outputs
output_path = Path.cwd() / "pyvale-output" / "dic_ex11"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)


# %%
# The argument that controls the output of the gray level residuals is
# ``output_residual``. The options are:
#
# * ``"NONE"``: no residuals are output (default).
# * ``"ABSOLUTE"``: outputs the absolute gray level residuals.
# * ``"NORMALIZED"``: outputs the normalized gray level residuals (recommended).

dic.calculate_2d(reference=data.dic_ex11_ref(),
                 deformed=data.dic_ex11_def(),
                 roi_mask=roi.mask,
                 seed=roi.seed,
                 subset_size=31,
                 subset_step=5,
                 partial_subset=0.5,
                 threshold=0.8,
                 max_displacement=10,
                 output_residual="NORMALIZED", # outputs the gray level residuals
                 output_basepath=output_path)

# %%
# The gray level residuals are outputted as a 2D array to the directory
# specified by ``output_basepath``. The GLR file with have the same prefix and delimiter as the DIC 
# results file. You'll be able to identify the GLR files as they'll have
# ``_residual.csv`` at the end of the filename. Below is an example of plotting the GLR
# overlaided on the deformed image.

# load the image and GLR
def_img = plt.imread(data.dic_ex11_def())
glr = np.loadtxt(output_path / "dic_results_ex11_def_residual.csv", delimiter=",")

plt.imshow(def_img, cmap="gray")
plt.imshow(glr,cmap="jet", alpha=0.6, vmin=0, vmax=0.8)
plt.show()

# %%
# .. image:: ../../../../_static/dic_ex11_results.png
#    :alt: Crack detection results
#    :width: 100%
#    :align: center
