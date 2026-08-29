import pytest

from pyvale.vfm.loadregimes import (
    LoadRegimeThresholds,
    resolve_load_regimes,
    resolve_relative_load_regimes,
)


def test_load_regimes_partition_frames_and_serialize():
    resolved = resolve_load_regimes([0.0, 0.05, 0.3, 0.8])
    assert resolved.pre_yield == (0,)
    assert resolved.onset == (1,)
    assert resolved.developed == (2,)
    assert resolved.late == (3,)
    assert resolved.diagnostics()["onset"] == [1]


def test_load_regime_thresholds_are_ordered():
    with pytest.raises(ValueError):
        LoadRegimeThresholds(onset=0.4, developed=0.3, late=0.8)


def test_relative_regimes_are_disjoint_and_follow_observed_progress():
    fractions = [
        0.0, 0.0, 0.0015, 0.0063, 0.0335, 0.0620, 0.0773,
        0.0898, 0.0991, 0.1076, 0.1159, 0.1307, 0.1477, 0.1625,
    ]
    resolved = resolve_relative_load_regimes(fractions)
    assert resolved.pre_yield == (0, 1, 2, 3)
    assert resolved.onset == (4, 5, 6)
    assert resolved.developed == (7, 8, 9, 10)
    assert resolved.late == (11, 12, 13)
    combined = resolved.pre_yield + resolved.onset + resolved.developed + resolved.late
    assert sorted(combined) == list(range(len(fractions)))
    assert len(set(combined)) == len(fractions)
    assert resolved.resolution == "relative_monotone_yield_progress"


def test_relative_regimes_regularise_sparse_crossings():
    resolved = resolve_relative_load_regimes([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert all(len(resolved.indices(name)) >= 2 for name in ("pre_yield", "onset", "developed", "late"))
