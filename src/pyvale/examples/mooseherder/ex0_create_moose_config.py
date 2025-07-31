# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Creating a configuration file
================================================================================

In this example we generate a json config file which help mooseherder find the
paths to moose apps we want to use in the future. To run these examples you will
need to have moose installed on your system. We use the proteus moose build
which you will find here: https://github.com/aurora-multiphysics/proteus. Or
you can build you own moose app: https://mooseframework.inl.gov/.
"""

#%%#
# We start with imports for paths and for our moose configuration object.
from pathlib import Path
from pyvale.mooseherder import MooseConfig

#%%
# A moose configuration is a dictionary with these three keys "main_path",
# "app_path" and "app_name". The first two keys retur resolved paths for moose
# and the app you are using. The last key returns the string used call your
# moose app on the command line. Once we have this dictionary we can build our
# moose config object which will perform some checks for us to see if the
# configuration is valid.
config = {'main_path': Path.home()/ 'moose',
          'app_path': Path.home() / 'proteus',
          'app_name': 'proteus-opt'}

moose_config = MooseConfig(config)

#%%
# We are going to save our moose configuration to the default output path for
# pyvale so if this default output path does not exist we create it. Then all we
# need to do is call the save configuration method and give it a path and file
# name for the json we want to save.
output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

config_path = Path.cwd() / 'moose-config.json'
moose_config.save_config(config_path)

#%%
# Now we have our moose configuration we can use it to run moose with
# mooseherder's mooserunner and mooseherd classes which we will see in later
# examples. For now we will move on to demonstrate how mooseherder modifies
# input files for gmsh and moose.



