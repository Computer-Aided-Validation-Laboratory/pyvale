import numpy as np
import pytest

from pyvale.vfm.residualfeatures import (
    coherent_rms,
    physical_length_to_odd_pixels,
    projected_rms,
    smooth_positive_part,
    weighted_cvar_abs,
    weighted_rms,
)


def test_weighted_rms_and_cvar_match_hand_calculation():
    values = np.array([1.0, 2.0, 3.0, 10.0])
    weights = np.full(4, 0.25)
    assert weighted_rms(values, weights).value == pytest.approx(np.sqrt(28.5))
    assert weighted_cvar_abs(values, weights, quantile=0.5).value == pytest.approx(6.5)


def test_cvar_is_monotone_when_largest_tail_value_increases():
    before = weighted_cvar_abs([0.0, 1.0, 2.0, 3.0], quantile=0.75).value
    after = weighted_cvar_abs([0.0, 1.0, 2.0, 4.0], quantile=0.75).value
    assert after > before


def test_coherent_rms_suppresses_alternating_noise_and_keeps_constant_signal():
    alternating = np.indices((31, 31)).sum(axis=0) % 2 * 2.0 - 1.0
    coherent = np.ones((31, 31))
    assert coherent_rms(alternating, sigma_pixels=2.0).value < 0.05
    assert coherent_rms(coherent, sigma_pixels=2.0).value == pytest.approx(1.0)


def test_coherent_rms_mask_normalisation_does_not_dilute_constant_field():
    values = np.ones((21, 21))
    values[:, :10] = np.nan
    assert coherent_rms(values, sigma_pixels=2.0).value == pytest.approx(1.0)


def test_projected_rms_agrees_with_direct_orthogonal_projection():
    values = np.array([1.0, 2.0, 3.0])
    basis = np.array([[1.0], [0.0], [0.0]])
    expected = 1.0 / np.sqrt(3.0)
    assert projected_rms(values, basis).value == pytest.approx(expected)


def test_physical_length_conversion_and_soft_positive_part():
    assert physical_length_to_odd_pixels(3.0, 0.2) == 15
    assert physical_length_to_odd_pixels(1.4, 0.2) == 7
    assert smooth_positive_part(-20.0, temperature=0.1) < 1e-10
    assert smooth_positive_part(2.0, temperature=0.1) == pytest.approx(2.0, rel=1e-9)
