from __future__ import annotations

import numpy as np

from pyvale.vfm.roi import build_roi_geometry, convert_mask_to_physical_roi


def test_mask_to_physical_roi_uses_support_cell_edges_not_sample_centres() -> None:
    x, y = np.meshgrid(np.array([10.0, 12.0, 14.0]), np.array([1.0, 4.0]))
    definition = convert_mask_to_physical_roi(np.ones(x.shape, dtype=bool), x, y, simplification_pixels=0.0)
    geometry = build_roi_geometry(definition)

    assert geometry.bounds == (9.0, -0.5, 15.0, 5.5)
    assert geometry.area == 36.0
