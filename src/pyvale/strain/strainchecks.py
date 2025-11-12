# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import os
import glob
from pathlib import Path


def check_strain_files(strain_files: str | Path) -> list[str]:
   
    filenames = []

    # Find deformation image files
    files = sorted(glob.glob(str(strain_files)))
    if not files:
        raise FileNotFoundError(f"No DIC data found: {strain_files}")

    for file in files:
        filenames.append(os.path.basename(file))

    return filenames
