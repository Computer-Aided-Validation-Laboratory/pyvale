# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Modifying gmsh input files
================================================================================

In this example we will use mooseherder's input modifier to programmatically
change variables in a gmsh .geo script.

We start by importing the packages we need for this example.
"""

from pathlib import Path

# pyvale imports
import pyvale.dataset as dataset
from pyvale.mooseherder import InputModifier

#%%
# We are going to use a gmsh geo file that is for a 2D rectangular plate with a
# hole in the center which we retrieve from pyvale's simulation library. We then
# use this to create an input modifier which has the correct comment string '//'
# for gmsh and the required line terminator ";".
gmsh_input = dataset.sim_case_gmsh_file_path(case_num=17)
gmsh_mod = InputModifier(gmsh_input, "//", ";")

#%%
# Note that the input modifier class only looks for variables between specified
# sentinel characters in comment lines which starts with '_*' and ends with '**'
# . An example variable block can be found in the gmsh input file.
#
# We then print the variables found in the gmsh input file to the console which
# are returned to us as a dictionary keyed by the variables string name in the
# file:
print(gmsh_mod.get_vars())
print()

#%%
# We can update the variables in the input modifier using a dictionary keyed by
# the variable names we want to change and the values being what we want to
# change them to. We do not have to use numeric values for these we can use
# expressions in strings.
new_vars = {"plate_width": 150e-3, "plate_height": "plate_width + 100e-3"}
gmsh_mod.update_vars(new_vars)

#%%
# Now we print the variables that are currently in the input modifier to check
# our modification worked.
print(gmsh_mod.get_vars())
print()

#%%
# Finally we want to save the modified input file to disk so we can run it with
# gmsh. First we create the standard pyvale-output directory so we can save the
# file there. Then we save the gmsh input file to the directory with a suitable
# name. Have a look at the file in the directory to ensure our modifications
# have worked.

output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

gmsh_save = output_path/"gmsh-mod-vars.geo"
gmsh_mod.write_file(gmsh_save)

