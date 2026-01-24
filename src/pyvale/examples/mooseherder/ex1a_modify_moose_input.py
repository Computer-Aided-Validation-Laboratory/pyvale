# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Modifying moose input files
================================================================================

In this example we will use mooseherder's input modifier to programmatically
change variables in a moose .i input script.

We start by importing the packages we need for this example.
"""

from pathlib import Path

# pyvale imports
import pyvale.dataset as dataset
from pyvale.mooseherder import InputModifier

#%%
# We use the input file for a simple thermo-mechanical cube with higher order
# quads. We pass this to our input modifier specifying the moose comment
# character(s) as '#' and moose input files do not have a line end character so
# we pass an empty string. In the next example we will modify a gmsh script
# where we will need to specify a line ending character.
moose_input = dataset.element_case_input_path(dataset.EElemTest.HEX20)
moose_mod = InputModifier(moose_input, comment_chars="#", end_chars="")

#%%
# Note that the input modifier class only looks for variables between specified
# sentinel characters in comment lines which starts with '_*' and ends with '**'
# . An example of what variable block can be found in the moose input file.
#
# We then print the variables found in the moose input file to the console which
# are returned to us as a dictionary keyed by the variables string name in the
# file:
print(moose_mod.get_vars())
print()

#%%
# We can update variables using the input modifier, we just need to create a
# dictionary keyed by the variable name we want to change and the value we want
# the variable to take. If the variable does not exist in the input file then an
# error will be raised. Here we will change the number of elements in 'x' and
# the elastic modulus.
new_vars = {"nElemX": 4, "EMod": 3.3e9}
moose_mod.update_vars(new_vars)


#%%
# Let's check that the variables in the dictionary have been updated to match
# what we have changed:
print(moose_mod.get_vars())
print()

#%%
# Now we want to save the modified input file so we first create the standard
# pyvale-output directory if it does not exist. Then we write the file to that
# directory with suitable file name. Have a look at the file in the
# pyvale-output directory to verify the variables have been changed.

output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

moose_save = output_path/"moose-mod-vars.i"
moose_mod.write_file(moose_save)

