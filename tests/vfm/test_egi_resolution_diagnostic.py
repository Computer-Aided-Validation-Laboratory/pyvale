from pathlib import Path
import sys

import numpy as np


DEV_VFM = Path(__file__).resolve().parents[2] / "dev/vfm"
if str(DEV_VFM) not in sys.path:
    sys.path.insert(0, str(DEV_VFM))

from egi_resolution_diagnostic import (  # noqa: E402
    derive_manual_supports,
    render_report,
    write_sweep_csv,
)


def test_diagnostic_support_bank_and_csv(tmp_path: Path) -> None:
    grid = np.linspace(0.0, 10.0, 101)
    x, y = np.meshgrid(grid, grid)
    fine, middle, broad = derive_manual_supports(x, y, 9)
    assert fine.window_size == (9, 9)
    assert middle.window_size == (21, 21)
    assert broad.window_size == (49, 49)
    rows = [{"support_pixels": 9, "support_mm": 0.9}]
    path = tmp_path / "sweep.csv"
    write_sweep_csv(path, rows)
    assert path.read_text().splitlines() == [
        "support_pixels,support_mm", "9,0.9",
    ]


def test_diagnostic_report_smoke(tmp_path: Path) -> None:
    x, y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    rows = [{
        "support_pixels": value,
        "support_mm": float(value),
        "informative_coverage": 1.0,
        "gated_egi_rms": 1.0 / value,
        "propagated_noise_rms": 0.1 / value,
        "diagnostic_snr": 10.0,
        "previous_support_correlation": np.nan if value == 3 else 0.9,
    } for value in range(3, 19, 2)]
    result = {
        "name": "smoke", "label": "clean reference noise", "rows": rows,
        "noise_knee_pixels": 7, "broad_pixels": 17, "runtime_seconds": 0.1,
        "representative_maps": {
            str(value): np.full((5, 5), value, dtype=float)
            for value in range(3, 19, 2)
        },
        "x": x, "y": y, "spatial_gate": np.ones((5, 5), dtype=bool),
    }
    report = render_report([result], tmp_path)
    assert report.exists()
    assert report.stat().st_size > 0
