"""Regression tests for the VFM gallery scripts."""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy.testing as npt
import pytest

import pyvale.vfm as vfm
from pyvale.examples.vfm.synthetic_rectangular_tensile import SyntheticTensileCase


@pytest.mark.parametrize(
    ("example_name", "expected_phase_count"),
    [
        ("vfm_ex1_hom", 1),
        ("vfm_ex2_slicewise", 1),
        ("vfm_ex3_slicewise_refinement", 1),
        ("vfm_ex4_hom_slicewise", 2),
    ],
)
def test_vfm_example_recovers_reference_parameters(
    example_name: str,
    expected_phase_count: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    original_backend = plt.get_backend()
    existing_figures = set(plt.get_fignums())
    switch_backend = original_backend.lower() != "agg"
    try:
        if switch_backend:
            plt.switch_backend("Agg")
        with plt.ioff():
            example_globals = runpy.run_module(f"pyvale.examples.vfm.{example_name}")
    finally:
        for figure_number in set(plt.get_fignums()) - existing_figures:
            plt.close(figure_number)
        if switch_backend:
            plt.switch_backend(original_backend)

    result = example_globals["result"]
    synthetic_case = example_globals["synthetic_case"]
    assert isinstance(result, vfm.IdentificationResult)
    assert isinstance(synthetic_case, SyntheticTensileCase)
    assert len(result.history.phases) == expected_phase_count
    assert result.history.phases[-1].solve_results[-1].success is True
    assert result.parameter_maps.keys() == synthetic_case.known_parameter_maps.keys()
    for parameter_name, reference_map in synthetic_case.known_parameter_maps.items():
        identified_map = result.parameter_maps[parameter_name]
        assert identified_map.shape == reference_map.shape
        npt.assert_allclose(
            identified_map,
            reference_map,
            rtol=5.0e-6,
            atol=1.0e-6,
            err_msg=parameter_name,
        )
