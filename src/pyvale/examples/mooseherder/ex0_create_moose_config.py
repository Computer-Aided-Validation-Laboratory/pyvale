# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from pathlib import Path
from pyvale.mooseherder import MooseConfig

config = {'main_path': Path.home()/ 'moose',
        'app_path': Path.home() / 'proteus',
        'app_name': 'proteus-opt'}

moose_config = MooseConfig(config)

save_path = Path.cwd() / 'moose-config.json'
moose_config.save_config(save_path)



