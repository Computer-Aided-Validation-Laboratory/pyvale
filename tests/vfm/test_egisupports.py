import numpy as np
import pytest

from pyvale.vfm.egisupports import (
    EgiSupportBankConfig,
    EgiSupportInformationSelectionConfig,
    EgiSupportSelectionConfig,
    EgiSignalSelectionConfig,
    analyse_egi_signal_sweep,
    analyse_egi_support_information,
    analyse_egi_support_sweep,
    generate_physical_egi_support_bank,
    generate_odd_pixel_egi_support_bank,
    resolve_physical_egi_supports,
    select_information_egi_supports,
    select_log_spaced_egi_supports,
    select_sparse_egi_supports,
)


def test_simple_signal_selector_uses_small_log_middle_and_broad() -> None:
    x, y = _regular_grid()
    supports = resolve_physical_egi_supports((0.6, 1.0, 1.4, 1.8), x, y)
    residuals = [np.full((2, *x.shape), value) for value in (0.5, 2.0, 3.0, 4.0)]
    noise = [np.ones_like(item) for item in residuals]
    sweep = analyse_egi_signal_sweep(
        supports, residuals, noise, active_fraction=0.25
    )

    selection = select_log_spaced_egi_supports(
        sweep,
        EgiSignalSelectionConfig(
            minimum_signal_to_noise=1.0,
            active_fraction=0.25,
        ),
    )

    assert selection.fine_index == 1
    assert selection.broad_index == 3
    assert selection.middle_index == 2
    assert selection.diagnostics(sweep)["log_middle_target"] == pytest.approx(
        np.sqrt(1.0 * 1.8)
    )


def test_simple_signal_selector_requires_three_resolved_scales() -> None:
    x, y = _regular_grid()
    supports = resolve_physical_egi_supports((0.6, 1.0, 1.8), x, y)
    residuals = [np.full(x.shape, value) for value in (0.1, 0.2, 2.0)]
    sweep = analyse_egi_signal_sweep(
        supports, residuals, [np.ones_like(x) for _ in supports]
    )

    with pytest.raises(ValueError, match="Fewer than three"):
        select_log_spaced_egi_supports(sweep)


def _regular_grid() -> tuple[np.ndarray, np.ndarray]:
    x_values = np.arange(7, dtype=np.float64) * 0.2
    y_values = np.arange(6, dtype=np.float64) * 0.2
    return np.meshgrid(x_values, y_values)


def test_physical_supports_are_resolved_and_pixel_duplicates_are_grouped() -> None:
    x, y = _regular_grid()

    supports = resolve_physical_egi_supports(
        (0.6, 1.0, 1.01, 1.8),
        x,
        y,
    )

    assert [support.window_size for support in supports] == [
        (3, 3),
        (5, 5),
        (9, 9),
    ]
    assert supports[1].requested_side_lengths == (1.0, 1.01)
    assert supports[1].nominal_side_lengths == pytest.approx((1.0, 1.0))
    assert supports[1].grid_spacing == pytest.approx((0.2, 0.2))


def test_support_sweep_records_noise_whitened_evidence_and_redundancy() -> None:
    x, y = _regular_grid()
    supports = resolve_physical_egi_supports((0.6, 1.0, 1.8), x, y)
    residuals = [np.full(x.shape, value) for value in (2.0, 3.0, 4.0)]
    residuals[0][0, 0] = np.nan
    noise = [np.array(2.0), np.array(1.0), np.array(4.0)]

    fine_response = np.zeros(x.shape)
    fine_response[:, :3] = 2.0
    middle_response = np.zeros(x.shape)
    middle_response[:, 3:] = 3.0
    broad_response = fine_response * 4.0

    sweep = analyse_egi_support_sweep(
        supports,
        residuals,
        noise,
        {"yield": (fine_response, middle_response, broad_response)},
    )

    assert sweep.evidence[0].valid_count == x.size - 1
    assert sweep.evidence[0].coverage_fraction == pytest.approx(
        (x.size - 1) / x.size
    )
    assert sweep.evidence[0].residual_rms == pytest.approx(2.0)
    assert sweep.evidence[0].whitened_residual_rms == pytest.approx(1.0)
    assert sweep.evidence[1].response_to_noise("yield") > 1.0
    assert sweep.absolute_cosine_redundancy[0, 2] == pytest.approx(1.0)
    assert sweep.absolute_cosine_redundancy[0, 1] == pytest.approx(0.0)


def test_sparse_selection_adds_an_independent_middle_support() -> None:
    x, y = _regular_grid()
    supports = resolve_physical_egi_supports((0.6, 1.0, 1.8), x, y)
    residuals = [np.ones(x.shape) for _ in supports]
    noise = [np.array(1.0) for _ in supports]
    fine_response = np.zeros(x.shape)
    fine_response[:, :3] = 2.0
    middle_response = np.zeros(x.shape)
    middle_response[:, 3:] = 3.0
    broad_response = fine_response * 2.0
    sweep = analyse_egi_support_sweep(
        supports,
        residuals,
        noise,
        {"yield": (fine_response, middle_response, broad_response)},
    )

    selection = select_sparse_egi_supports(
        sweep,
        EgiSupportSelectionConfig(
            minimum_response_to_noise=0.5,
            maximum_middle_redundancy=0.8,
        ),
    )

    assert selection.selected_indices == (0, 1, 2)
    assert dict(selection.roles) == {"fine": 0, "broad": 2, "middle": 1}


def test_sparse_selection_rejects_a_bank_below_the_information_gate() -> None:
    x, y = _regular_grid()
    supports = resolve_physical_egi_supports((0.6,), x, y)
    sweep = analyse_egi_support_sweep(
        supports,
        (np.ones(x.shape),),
        (np.array(1.0),),
        {"yield": (np.zeros(x.shape),)},
    )

    with pytest.raises(ValueError, match="No EGI support"):
        select_sparse_egi_supports(sweep)


def test_candidate_bank_uses_physical_extents_and_deduplicates_windows() -> None:
    x = np.tile(np.arange(31, dtype=np.float64) * 0.2, (21, 1))
    y = np.tile((np.arange(21, dtype=np.float64) * 0.1)[:, np.newaxis], (1, 31))

    supports = generate_physical_egi_support_bank(
        x, y, EgiSupportBankConfig(candidate_count=10)
    )

    assert supports[0].window_size == (7, 3)
    assert supports[-1].nominal_side_length <= 1.1
    assert all(
        current.nominal_side_length < following.nominal_side_length
        for current, following in zip(supports, supports[1:])
    )


def test_minimal_bank_contains_every_odd_pixel_window() -> None:
    x, y = np.meshgrid(np.arange(21, dtype=float), np.arange(21, dtype=float))

    supports = generate_odd_pixel_egi_support_bank(x, y)

    assert [item.window_size for item in supports] == [
        (3, 3), (5, 5), (7, 7), (9, 9)
    ]


def test_information_selector_uses_multi_probe_fisher_gain_for_middle() -> None:
    x, y = _regular_grid()
    supports = resolve_physical_egi_supports((0.6, 1.0, 1.4, 1.8), x, y)
    residuals = [np.ones(x.shape) for _ in supports]
    noise = [np.ones(x.shape) for _ in supports]
    # Fine and broad both observe probe 0.  The middle support is the only
    # candidate that contributes the complementary local probe 1 direction.
    response_fields = []
    for index in range(len(supports)):
        responses = np.zeros((3, *x.shape))
        responses[0] = 2.0  # homogeneous yield
        if index == 1:
            responses[1] = 0.2
            responses[2] = 4.0
        elif index in {0, 3}:
            responses[1] = 2.0
            responses[2] = 0.1
        else:
            responses[1] = 2.0
            responses[2] = 0.2
        response_fields.append(responses)

    sweep = analyse_egi_support_information(
        supports,
        residuals,
        noise,
        response_fields,
        probe_names=("homogeneous_yield", "local_yield_a", "local_yield_b"),
    )
    selection = select_information_egi_supports(
        sweep,
        EgiSupportInformationSelectionConfig(
            minimum_response_to_noise=1.0,
            minimum_local_probe_fraction=0.5,
        ),
    )

    assert selection.fine_index == 0
    assert selection.broad_index == 3
    assert selection.middle_index == 1
    assert selection.status == "three_resolved"
    assert selection.middle_information_gain is not None
    assert selection.middle_information_gain > 0.0


def test_information_selector_requires_declared_local_and_homogeneous_probes() -> None:
    x, y = _regular_grid()
    supports = resolve_physical_egi_supports((0.6, 1.8), x, y)
    response_fields = [np.ones((1, *x.shape)) for _ in supports]
    sweep = analyse_egi_support_information(
        supports,
        [np.ones(x.shape) for _ in supports],
        [np.ones(x.shape) for _ in supports],
        response_fields,
        probe_names=("wrong_name",),
    )

    with pytest.raises(ValueError, match="local yield"):
        select_information_egi_supports(sweep)
