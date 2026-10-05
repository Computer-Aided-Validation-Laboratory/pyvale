# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ==============================================================================

from pathlib import Path
import numpy as np

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.valid as val

data_path = dataset.valid_data_dir()

load_opts = io.LoadOpts(delimiter=",",header_rows=0,)

