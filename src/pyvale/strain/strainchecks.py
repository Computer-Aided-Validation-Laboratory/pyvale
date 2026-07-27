# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import os
import glob
from pathlib import Path


def check_strain_files(strain_files: str | Path | list[Path] | list[str]) -> list[str]:
    """
    Check for strain/deformation files in the given path(s) and return their filenames.

    Parameters
    ----------
    strain_files : str, pathlib.Path, list[str], or list[Path]
        Path(s) or glob pattern(s) pointing to the strain/deformation files.

    Returns
    -------
    list[str]
        A sorted list of filenames (not full paths) matching the input path(s)/pattern(s).

    Raises
    ------
    FileNotFoundError
        If no files matching the given path(s) or pattern(s) are found.

    Examples
    --------
    >>> check_strain_files("data/strain_*.tif")
    ['strain_001.tif', 'strain_002.tif', 'strain_003.tif']
    """
    if isinstance(strain_files, (str, Path)):
        patterns = [strain_files]
    else:
        patterns = strain_files

    files = []
    for pattern in patterns:
        files.extend(glob.glob(str(pattern)))

    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"No DIC data found: {strain_files}")

    return [os.path.basename(file) for file in files]
