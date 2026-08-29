import pytest

from pyvale.vfm.loadregimes import (
    LoadRegimeThresholds,
    resolve_load_regimes,
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
