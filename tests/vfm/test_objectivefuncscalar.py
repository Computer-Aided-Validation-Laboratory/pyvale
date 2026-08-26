import numpy as np
import pytest

from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefuncscalar import ScalarFirstResultRms


def test_scalar_first_result_rms_ignores_nonfinite_values() -> None:
    objective = ScalarFirstResultRms()

    result = objective.evaluate(
        [MetricResult(residual=np.array([3.0, 4.0, np.nan]))]
    )

    assert result == pytest.approx(np.sqrt(12.5))


def test_scalar_first_result_rms_requires_finite_residual() -> None:
    with pytest.raises(ValueError, match="finite"):
        ScalarFirstResultRms().evaluate(
            [MetricResult(residual=np.array([np.nan]))]
        )
