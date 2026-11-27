# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================

"""Compare each packaged demo with the same scene routed through PyVale."""

from __future__ import annotations

import filecmp
import importlib
import shutil
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import riley
from riley.__main__ import _DEMO_FUNCS, _PROCEDURAL_DEMOS

from pyvale import render


@pytest.fixture
def demo_directories(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """Provide isolated working directories and clean up large demo outputs."""
    native_dir = tmp_path / "native"
    pyvale_dir = tmp_path / "pyvale"
    native_dir.mkdir()
    pyvale_dir.mkdir()
    yield native_dir, pyvale_dir
    shutil.rmtree(native_dir, ignore_errors=True)
    shutil.rmtree(pyvale_dir, ignore_errors=True)


@pytest.mark.riley
@pytest.mark.parametrize("demo_name", sorted(_DEMO_FUNCS))
def test_packaged_demo_parity(
    demo_directories: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    demo_name: str,
) -> None:
    """Compare render files, calibration, metadata and any returned arrays."""
    native_dir, pyvale_dir = demo_directories
    demo = importlib.import_module(_DEMO_FUNCS[demo_name])
    args = ([],) if demo_name in _PROCEDURAL_DEMOS else ()
    raster = riley.raster
    native_images: deque[np.ndarray | None] = deque()

    def record_native(
        meshes: riley.Mesh | list[riley.Mesh] | tuple[riley.Mesh, ...],
        cameras: riley.Camera | list[riley.Camera] | tuple[riley.Camera, ...],
        config: riley.RasterConfig,
        out_dir: str | None = None,
    ) -> np.ndarray | None:
        images = raster(meshes, cameras, config, out_dir)
        native_images.append(images)
        return images

    def render_through_pyvale(
        meshes: riley.Mesh | list[riley.Mesh] | tuple[riley.Mesh, ...],
        cameras: riley.Camera | list[riley.Camera] | tuple[riley.Camera, ...],
        config: riley.RasterConfig,
        out_dir: str | None = None,
    ) -> np.ndarray | None:
        scene_meshes = [meshes] if isinstance(meshes, riley.Mesh) else list(meshes)
        scene_cameras = [cameras] if isinstance(cameras, riley.Camera) else list(cameras)
        scene = render.Scene3D(scene_meshes, scene_cameras)
        output_path = Path(out_dir) if out_dir is not None else None
        with monkeypatch.context() as backend:
            backend.setattr(riley, "raster", raster)
            result = render.Riley(config, output_path).render(scene)
        expected = native_images.popleft()
        if expected is None:
            assert result.images is None
            return None
        assert result.images is not None
        assert result.images.dtype == expected.dtype
        np.testing.assert_array_equal(result.images, expected.transpose(1, 0, 3, 4, 2))
        # Preserve the layout the rest of the upstream demo expects.
        return result.images.transpose(1, 0, 4, 2, 3)

    with monkeypatch.context() as run:
        run.chdir(native_dir)
        run.setattr(riley, "raster", record_native)
        demo.main(*args)

        assert native_images, f"{demo_name} did not call riley.raster"

        run.chdir(pyvale_dir)
        run.setattr(riley, "raster", render_through_pyvale)
        demo.main(*args)

    assert not native_images, f"{demo_name} skipped a native raster call"

    native_files = sorted(
        path.relative_to(native_dir) for path in native_dir.rglob("*") if path.is_file()
    )
    pyvale_files = sorted(
        path.relative_to(pyvale_dir) for path in pyvale_dir.rglob("*") if path.is_file()
    )
    assert native_files, f"{demo_name} produced no files"
    assert native_files == pyvale_files
    for relative_path in native_files:
        assert filecmp.cmp(
            native_dir / relative_path, pyvale_dir / relative_path, shallow=False,
        ), f"render output differs: {relative_path}"
