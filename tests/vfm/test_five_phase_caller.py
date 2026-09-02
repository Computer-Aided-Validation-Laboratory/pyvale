from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


DEV_VFM = Path(__file__).resolve().parents[2] / "dev/vfm"
if str(DEV_VFM) not in sys.path:
    sys.path.insert(0, str(DEV_VFM))

import call_notched_ebw_five_phase_identification as five_phase_runner


@pytest.mark.parametrize("hardening_fixed", [False, True])
def test_phase_2_uses_configured_hardening_status(
    monkeypatch, tmp_path, hardening_fixed
) -> None:
    captured = {}
    expected = (object(), [1.0, 1.0])

    def build_phase(args, experiment, output_dir, *, hardening_fixed):
        captured.update({
            "args": args,
            "experiment": experiment,
            "output_dir": output_dir,
            "hardening_fixed": hardening_fixed,
        })
        return expected

    monkeypatch.setattr(
        five_phase_runner.phase_2_runner,
        "_create_spatial_yield_phase",
        build_phase,
    )
    args = SimpleNamespace(phase_2_fix_hardening=hardening_fixed)
    experiment = object()

    actual = five_phase_runner._create_phase_2(args, experiment, tmp_path)

    assert actual is expected
    assert captured == {
        "args": args,
        "experiment": experiment,
        "output_dir": tmp_path,
        "hardening_fixed": hardening_fixed,
    }


def test_sbvf_reconciliation_projects_additive_maps_to_physical_bounds() -> None:
    phase = five_phase_runner._sbvf_phase({}, max_evaluations=10)

    expected = {
        "yield_strength": five_phase_runner.phase_2_runner.YIELD_BOUNDS_MPA,
        "hardening_modulus": five_phase_runner.phase_2_runner.HARDENING_BOUNDS_MPA,
    }
    assert phase.optimiser.parameter_map_bounds == expected
    assert phase.metrics[0].parameter_map_bounds == expected
