# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

"""Utilities for reporting the Python and compiled pyvale environment."""

from __future__ import annotations

import importlib.util
import importlib
import platform
import sys
from pathlib import Path
from typing import Any


def build_info() -> dict[str, Any]:
    """Return information about the installed pyvale build and runtime.

    The function reports the location of available compiled modules and reads
    Git metadata from the common compiled extension. It remains independent of
    the DIC, calibration, and strain extensions.
    """

    import pyvale

    compiled_modules: dict[str, str | None] = {}
    for module_name in (
        "pyvale.commoncpp.commoncpp",
        "pyvale.dic.diccpp",
        "pyvale.calib.calibcpp",
        "pyvale.strain.strain_cpp",
    ):
        try:
            module_spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            module_spec = None

        compiled_modules[module_name] = (
            str(Path(module_spec.origin).resolve())
            if module_spec is not None and module_spec.origin is not None
            else None
        )

    git_commit: str | None = None
    git_branch: str | None = None
    git_clean: bool | None = None
    try:
        commoncpp = importlib.import_module("pyvale.commoncpp.commoncpp")
        git_commit = getattr(commoncpp, "_git_commit", None)
        git_branch = getattr(commoncpp, "_git_branch", None)
        git_clean = getattr(commoncpp, "_git_clean", None)
    except (ImportError, ModuleNotFoundError):
        pass

    return {
        "pyvale_version": pyvale.__version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "package_path": str(Path(pyvale.__file__).resolve().parent),
        "compiled_modules": compiled_modules,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_clean": git_clean,
    }
