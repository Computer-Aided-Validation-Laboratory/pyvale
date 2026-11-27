# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Windows headless routing without platform mutation or a native Windows driver."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from pyvale.sensorsim.visualbackend import _configure_windows_headless_backend


def test_windows_headless_selects_osmesa_and_restores_interactive_backend() -> None:
    environment = {}
    loaded = []

    def load_library(name: str) -> object:
        loaded.append(name)
        return object()

    with _configure_windows_headless_backend(
        platform="win32",
        off_screen=True,
        out_environment=environment,
        load_library=load_library,
    ):
        assert environment["VTK_DEFAULT_OPENGL_WINDOW"] == "vtkOSOpenGLRenderWindow"
        assert environment["LP_NUM_THREADS"] == "0"
    assert loaded == ["osmesa.dll"]
    assert "VTK_DEFAULT_OPENGL_WINDOW" not in environment
    # Mesa initialises at render time, after the plotter constructor returns.
    assert environment["LP_NUM_THREADS"] == "0"


@pytest.mark.parametrize(
    "platform, off_screen, override",
    [
        ("linux", True, None),
        ("darwin", True, None),
        ("win32", False, None),
        ("win32", True, "vtkWin32OpenGLRenderWindow"),
    ],
)
def test_other_modes_and_explicit_backend_choices_are_preserved(
    platform: str,
    off_screen: bool,
    override: str | None,
) -> None:
    environment = {} if override is None else {"VTK_DEFAULT_OPENGL_WINDOW": override}
    original = environment.copy()

    def reject_library_load(name: str) -> object:
        raise AssertionError(f"Unexpected OSMesa load: {name}")

    with _configure_windows_headless_backend(
        platform=platform,
        off_screen=off_screen,
        out_environment=environment,
        load_library=reject_library_load,
    ):
        assert environment == original
    assert environment == original


def test_missing_mesa_raises_clear_error_before_native_rendering() -> None:
    environment = {}

    def missing_library(name: str) -> object:
        raise OSError(f"Cannot load {name}")

    with pytest.raises(RuntimeError, match="osmesa.dll"):
        with _configure_windows_headless_backend(
            platform="win32",
            off_screen=True,
            out_environment=environment,
            load_library=missing_library,
        ):
            raise AssertionError("Rendering must not start without Mesa.")
    assert environment == {}


def test_constructor_error_restores_backend_and_keeps_explicit_thread_setting() -> None:
    environment = {"VTK_DEFAULT_OPENGL_WINDOW": "", "LP_NUM_THREADS": "2"}
    original = environment.copy()

    def load_library(name: str) -> object:
        assert name == "osmesa.dll"
        return object()

    with pytest.raises(ValueError, match="constructor"):
        with _configure_windows_headless_backend(
            platform="win32",
            off_screen=True,
            out_environment=environment,
            load_library=load_library,
        ):
            assert environment["LP_NUM_THREADS"] == "2"
            assert environment["VTK_DEFAULT_OPENGL_WINDOW"] == "vtkOSOpenGLRenderWindow"
            raise ValueError("constructor failed")
    assert environment == original


def test_explicit_osmesa_backend_is_checked_and_preserved() -> None:
    environment = {"VTK_DEFAULT_OPENGL_WINDOW": "vtkOSOpenGLRenderWindow"}
    loaded = []

    def load_library(name: str) -> object:
        loaded.append(name)
        return object()

    with _configure_windows_headless_backend(
        platform="win32",
        off_screen=True,
        out_environment=environment,
        load_library=load_library,
    ):
        assert loaded == ["osmesa.dll"]
    assert environment["VTK_DEFAULT_OPENGL_WINDOW"] == "vtkOSOpenGLRenderWindow"


def test_concurrent_interactive_plotter_cannot_inherit_headless_override() -> None:
    environment = {}
    headless_entered = Event()
    interactive_started = Event()
    interactive_entered = Event()
    release_headless = Event()

    def load_library(name: str) -> object:
        assert name == "osmesa.dll"
        return object()

    def construct_plotter(*, off_screen: bool) -> str | None:
        if not off_screen:
            interactive_started.set()
        with _configure_windows_headless_backend(
            platform="win32",
            off_screen=off_screen,
            out_environment=environment,
            load_library=load_library,
        ):
            if off_screen:
                headless_entered.set()
                if not release_headless.wait(timeout=5):
                    raise TimeoutError("Headless construction was not released.")
            else:
                interactive_entered.set()
            return environment.get("VTK_DEFAULT_OPENGL_WINDOW")

    with ThreadPoolExecutor(max_workers=2) as executor:
        headless = executor.submit(construct_plotter, off_screen=True)
        assert headless_entered.wait(timeout=5)
        interactive = executor.submit(construct_plotter, off_screen=False)
        try:
            assert interactive_started.wait(timeout=5)
            assert not interactive_entered.wait(timeout=0.1)
        finally:
            release_headless.set()
        assert headless.result(timeout=5) == "vtkOSOpenGLRenderWindow"
        assert interactive.result(timeout=5) is None
    assert "VTK_DEFAULT_OPENGL_WINDOW" not in environment
