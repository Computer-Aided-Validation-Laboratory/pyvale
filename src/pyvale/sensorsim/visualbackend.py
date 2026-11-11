# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Internal Windows backend selection for off-screen PyVista plotters."""

import ctypes
import os
import sys
from collections.abc import Callable, Iterator, MutableMapping
from contextlib import contextmanager
from threading import RLock
from typing import Final

OSMESA_BACKEND: Final[str] = "vtkOSOpenGLRenderWindow"
_RENDER_WINDOW_LOCK: Final = RLock()


@contextmanager
def _configure_windows_headless_backend(
    *,
    off_screen: bool,
    platform: str = sys.platform,
    out_environment: MutableMapping[str, str] | None = None,
    load_library: Callable[[str], object] = ctypes.CDLL,
) -> Iterator[None]:
    """Choose OSMesa while constructing a Windows off-screen render window.

    A caller's explicit backend takes precedence. The factory setting is restored
    afterward so subsequent interactive windows retain their normal backend.
    Mesa's default LP_NUM_THREADS=0 is retained for the process: the rasterizer
    initialises at first render, and its worker teardown can hang on Windows.
    Mesa DLL installation belongs to the host environment, not this module.

    The private dependency arguments allow routing/error checks on other systems
    without modifying platform globals or loading a Windows DLL there.
    """
    if platform != "win32":
        yield
        return
    # Environment selection is process-wide. Include interactive construction in
    # the lock so another PyVale plotter cannot inherit the temporary override.
    with _RENDER_WINDOW_LOCK:
        if not off_screen:
            yield
            return
        environment = os.environ if out_environment is None else out_environment
        previous_backend = environment.get("VTK_DEFAULT_OPENGL_WINDOW")
        if previous_backend not in (None, "", OSMESA_BACKEND):
            yield
            return
        previous_threads = environment.get("LP_NUM_THREADS")
        environment.setdefault("LP_NUM_THREADS", "0")
        try:
            # Keep the DLL loaded while VTK constructs its render window.
            _osmesa = load_library("osmesa.dll")
        except OSError as error:
            if previous_threads is None:
                environment.pop("LP_NUM_THREADS", None)
            raise RuntimeError(
                "Windows headless rendering requires Mesa's osmesa.dll and its "
                "dependencies on the DLL search path. Install the Mesa3D off-screen "
                "renderer; GitHub Actions can use "
                "pyvista/setup-headless-display-action "
                "with install-mesa3d-offscreen enabled."
            ) from error
        environment["VTK_DEFAULT_OPENGL_WINDOW"] = OSMESA_BACKEND
        try:
            yield
        finally:
            if previous_backend is None:
                environment.pop("VTK_DEFAULT_OPENGL_WINDOW", None)
            else:
                environment["VTK_DEFAULT_OPENGL_WINDOW"] = previous_backend
