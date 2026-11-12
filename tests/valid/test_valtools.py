# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Analytic conditional ECDFs with separate epistemic and aleatory axes."""

import numpy as np
import pytest

import pyvale.valid as val


@pytest.fixture
def quantity() -> val.ValArray:
    return val.ValArray(
        np.array(
            [
                [[0.0, 0.0, 2.0], [2.0, 2.0, 4.0]],
                [[10.0, 10.0, 12.0], [12.0, 12.0, 14.0]],
            ]
        ),
        (
            val.EValAxis.EPISTEMIC_TRANS,
            val.EValAxis.EPISTEMIC_NUIS,
            val.EValAxis.ALEATORY,
        ),
        offsets=val.EpistemicOffsets(lower=1.0, upper=2.0),
    )


def test_extract_ecdf_selects_epistemic_cell_and_retains_duplicate_mass(
    quantity: val.ValArray,
) -> None:
    cdf = val.extract_ecdf(
        quantity,
        epis_trans_idx=1,
        epis_nuis_idx=0,
        offset=val.EEpisOffset.UPPER,
    )
    np.testing.assert_array_equal(cdf.quantiles, [12.0, 14.0])
    np.testing.assert_allclose(cdf.probabilities, [2.0 / 3.0, 1.0])
    assert quantity.data[1, 0, 0] == 10.0


def test_envelope_fixes_transferable_axis_and_spans_nuisance_axis(
    quantity: val.ValArray,
) -> None:
    support, lower, upper = val.extract_ecdf_envelope(
        quantity,
        epis_trans_idx=1,
        offset=val.EEpisOffset.NOMINAL,
    )
    np.testing.assert_array_equal(support, [10.0, 12.0, 14.0])
    np.testing.assert_allclose(lower, [0.0, 2.0 / 3.0, 1.0])
    np.testing.assert_allclose(upper, [2.0 / 3.0, 1.0, 1.0])


def test_envelope_spans_coherent_offsets_of_one_epistemic_cell(
    quantity: val.ValArray,
) -> None:
    support, lower, upper = val.extract_ecdf_envelope(
        quantity,
        epis_trans_idx=1,
        epis_nuis_idx=0,
    )
    np.testing.assert_array_equal(support, [9.0, 10.0, 11.0, 12.0, 14.0])
    np.testing.assert_allclose(lower, [0.0, 0.0, 0.0, 2.0 / 3.0, 1.0])
    np.testing.assert_allclose(upper, [2.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0])


def test_calc_ecdf_omits_nan() -> None:
    cdf = val.calc_ecdf(np.array([3.0, np.nan, 1.0, 1.0]))
    np.testing.assert_array_equal(cdf.quantiles, [1.0, 3.0])
    np.testing.assert_allclose(cdf.probabilities, [2.0 / 3.0, 1.0])


@pytest.mark.parametrize(
    "samples", [np.ones((2, 2)), np.array([np.inf]), np.array([np.nan]), np.array([])]
)
def test_calc_ecdf_rejects_invalid_samples(samples: np.ndarray) -> None:
    with pytest.raises(ValueError):
        val.calc_ecdf(samples)


@pytest.mark.parametrize("index", [-1, 2])
def test_extraction_rejects_invalid_epistemic_indices(
    quantity: val.ValArray,
    index: int,
) -> None:
    with pytest.raises(IndexError, match="EPISTEMIC_TRANS"):
        val.extract_ecdf(quantity, epis_trans_idx=index)
    with pytest.raises(IndexError, match="EPISTEMIC_TRANS"):
        val.extract_ecdf_envelope(quantity, epis_trans_idx=index)


def test_extraction_rejects_nonzero_index_on_absent_axis() -> None:
    quantity = val.ValArray(np.array([0.0, 1.0]), (val.EValAxis.ALEATORY,))
    with pytest.raises(IndexError, match="EPISTEMIC_NUIS"):
        val.extract_ecdf(quantity, epis_nuis_idx=1)
