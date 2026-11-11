# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Parity with the independently executed fullfieldvalmetrics Pulse255 analysis."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import pyvale.data as dataset
import pyvale.valid as val


@dataclass(slots=True)
class Pulse25XReference:
    simulation: np.ndarray
    sensor_tags: tuple[str, ...]
    experiment: dict[str, np.ndarray]
    offsets: dict[str, float]
    expected: pd.DataFrame


@pytest.fixture(scope="module")
def reference() -> Pulse25XReference:
    data_path = dataset.valid_data_dir()
    with np.load(data_path / "pulse25x_full_simulation.npz") as saved:
        simulation = saved["simulation"]
        tags = tuple(saved["sensor_tags"].tolist())
    experiment = {}
    for suffix, channels in (
        (
            "SteadyDICData",
            ("TC1", "TC3", "TC4", "TC5", "TC6", "TC7", "TC8", "TC9", "TC10"),
        ),
        ("SteadyHIVEData", ("TC2",)),
        ("SteadyPICOData", ("CV",)),
    ):
        frame = pd.read_csv(data_path / f"Pulse255_{suffix}.csv")
        for channel in channels:
            key = "Coil RMS Voltage" if channel == "CV" else channel
            experiment[channel] = frame[key].to_numpy(dtype=float)
    # Reproduce the original positional association, not a corrected tag lookup.
    frame = pd.read_csv(data_path / "pulse25x_uncertainty_summary.csv")
    uncertainty = frame.iloc[-1, -11:].to_numpy(dtype=float)
    uncertainty = np.r_[uncertainty[1:], uncertainty[0]]
    offsets = dict(zip(experiment, uncertainty, strict=True))
    expected = pd.read_csv(data_path / "pulse25x_reference_mavm.csv", index_col=0)
    return Pulse25XReference(simulation, tags, experiment, offsets, expected)


@pytest.mark.parametrize(
    "channel", ("TC3", "TC5", "TC6", "TC8", "TC9", "TC10", "TC2", "CV")
)
def test_all_paper_endpoint_distances_match_independent_reference(
    reference: Pulse25XReference,
    channel: str,
) -> None:
    scenarios = reference.simulation[reference.sensor_tags.index(channel)]
    support_sums = np.sum(np.sort(scenarios, axis=1), axis=1)
    lower_index = int(np.argmin(support_sums))
    upper_index = int(np.argmax(support_sums))
    experimental_support = np.unique(reference.experiment[channel])
    for exp_name, sign in (("min", -1.0), ("max", 1.0)):
        experiment = experimental_support + sign * reference.offsets[channel]
        for sim_name, index in (("min", lower_index), ("max", upper_index)):
            calculated = val.calc_mavm_1d(
                scenarios[index], experiment, duplicates=val.EMAVMDuplicates.REJECT
            )
            for direction, distance in (
                ("d+", calculated.d_plus),
                ("d-", calculated.d_minus),
            ):
                row = f"sim-{sim_name}_exp-{exp_name}-{direction}"
                assert distance == pytest.approx(
                    reference.expected.loc[row, channel], rel=1e-12, abs=1e-12
                )
    correction = val.calc_mavm_correction(
        scenarios[lower_index],
        scenarios[upper_index],
        experimental_support - reference.offsets[channel],
        experimental_support + reference.offsets[channel],
        duplicates=val.EMAVMDuplicates.REJECT,
    )
    expected_lower = max(
        reference.expected.loc[f"sim-min_exp-{exp_name}-d-", channel]
        for exp_name in ("min", "max")
    )
    expected_upper = max(
        reference.expected.loc[f"sim-max_exp-{exp_name}-d+", channel]
        for exp_name in ("min", "max")
    )
    assert correction.lower.d_minus == pytest.approx(
        expected_lower, rel=1e-12, abs=1e-12
    )
    assert correction.upper.d_plus == pytest.approx(
        expected_upper, rel=1e-12, abs=1e-12
    )


def test_reference_retains_missing_channels_and_documents_duplicate_weighting(
    reference: Pulse25XReference,
) -> None:
    assert reference.simulation.shape == (11, 250, 400)
    for channel in ("TC1", "TC4", "TC7"):
        assert np.all(np.isnan(reference.experiment[channel]))
    assert len(reference.experiment["TC2"]) == 447
    assert len(np.unique(reference.experiment["TC2"])) == 311
    assert len(reference.experiment["CV"]) == 847
    assert len(np.unique(reference.experiment["CV"])) == 817
