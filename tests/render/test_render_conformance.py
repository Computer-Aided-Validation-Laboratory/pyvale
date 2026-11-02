"""Cross-backend rendering conformance gold regressions."""

from pathlib import Path

import numpy as np
import pytest

import pyvale.render as render
from pyvale.verif.renderconformance import (
    IMAGE_SIZE,
    RenderConformanceCase,
    conformance_cases,
    feebee_scene,
    render_backend_case,
)
from pyvale.verif.renderverif import assert_render_allclose


GOLD_ROOT = Path(__file__).parent
BACKENDS = (
    "blender",
    "riley",
)


@pytest.mark.parametrize(
    "case",
    conformance_cases(),
    ids=lambda case: case.name,
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_render_backend_conformance_gold(
    backend: str,
    case: RenderConformanceCase,
    tmp_path: Path,
) -> None:
    """Every functioning backend renders the canonical deforming Tri3 cases."""
    if backend == "blender" and not render.blender_available():
        pytest.skip("The optional Blender renderer backend is unavailable.")

    actual = render_backend_case(backend, case, tmp_path / backend / case.name)
    expected = np.load(
        GOLD_ROOT / f"gold_{backend}" / f"{case.name}.npy",
    )

    assert actual.shape == (2, 1, IMAGE_SIZE, IMAGE_SIZE, 1)
    assert np.isfinite(actual).all()
    assert not np.array_equal(actual[0], actual[1])

    tolerance = 2.0 if backend == "blender" else 1.0e-9
    assert_render_allclose(
        actual,
        expected,
        f"{backend}_{case.name}",
        rtol=0.0,
        atol=tolerance,
    )


@pytest.mark.parametrize(
    "case",
    conformance_cases(),
    ids=lambda case: case.name,
)
def test_feebee_conformance_scene_reaches_backend_boundary(
    case: RenderConformanceCase,
) -> None:
    """Feebee accepts every case while its compiled backend is unavailable."""
    renderer = render.Feebee()
    scene = feebee_scene(case)

    renderer.verify_input(scene)
    with pytest.raises(NotImplementedError, match="compiled rendering backend"):
        renderer._render(scene)
