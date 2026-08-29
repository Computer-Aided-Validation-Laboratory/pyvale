import importlib.util
from pathlib import Path
import sys

import numpy as np


_PATH = Path(__file__).parents[2] / "dev/vfm/build_notched_ebw_direct_fit_reference.py"
_SPEC = importlib.util.spec_from_file_location("direct_fit_reference", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_direct_fit_error_metrics_are_zero_for_truth():
    truth = np.arange(16, dtype=float).reshape(4, 4) + 1.0
    roi = np.ones_like(truth, dtype=bool)
    yielded = truth < 15.0
    high = truth < 5.0
    errors = _MODULE._errors(truth, truth.copy(), roi, yielded, high)
    assert all(value == 0.0 for value in errors.values())
