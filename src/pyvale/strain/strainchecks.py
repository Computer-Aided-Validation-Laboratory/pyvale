# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import os
import glob
from pathlib import Path


def check_strain_files(strain_files: str | Path) -> list[str]:
    """
    Check for strain/deformation files in the given path and return their filenames.

    Parameters
    ----------
    strain_files : str or pathlib.Path
        Path or glob pattern pointing to the strain/deformation files.

    Returns
    -------
    list[str]
        A sorted list of filenames (not full paths) matching the input path/pattern.

    Raises
    ------
    FileNotFoundError
        If no files matching the given path or pattern are found.

    Examples
    --------
    >>> check_strain_files("data/strain_*.tif")
    ['strain_001.tif', 'strain_002.tif', 'strain_003.tif']
    """

    filenames = []

    # Find deformation image files
    files = sorted(glob.glob(str(strain_files)))
    if not files:
        raise FileNotFoundError(f"No DIC data found: {strain_files}")

    for file in files:
        filenames.append(os.path.basename(file))

    return filenames
