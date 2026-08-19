"""Tests for render image-regression failure diagnostics."""

import numpy as np
import pytest

from render_checks import assert_render_allclose


def test_failed_image_comparison_writes_all_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A mismatch creates scaled TIFF and raw NumPy diagnostic images."""
    monkeypatch.chdir(tmp_path)
    reference = np.zeros((4, 4))
    actual = np.ones((4, 4))

    with pytest.raises(AssertionError, match="Render mismatch"):
        assert_render_allclose(actual, reference, "test mismatch")

    directory = tmp_path / "render-fails/test_mismatch"
    for name in (
        "render.npy", "reference.npy", "difference.npy", "render.tiff",
        "reference.tiff", "difference.tiff",
    ):
        assert (directory / name).is_file()
