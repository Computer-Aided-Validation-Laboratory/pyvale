# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Smoke tests for the basic and extended Sensorsim gallery examples."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


_BASIC_EXAMPLES = (
    "basicsensorsim/ex0_quickstart.py",
    "basicsensorsim/ex1_scalar_sensors.py",
    "basicsensorsim/ex2_vector_tensor_sensors.py",
    "basicsensorsim/ex3_experiment_simulator.py",
)

_EXTENDED_EXAMPLES = (
    "extsensorsim/ex1_byosimdata.py",
    "extsensorsim/ex2_meshfreesensors.py",
    "extsensorsim/ex3a_scal2d.py",
    "extsensorsim/ex3b_scal3d.py",
    "extsensorsim/ex3c_vec2d.py",
    "extsensorsim/ex3d_vec3d.py",
    "extsensorsim/ex3e_tens2d.py",
    "extsensorsim/ex3f_tens3d.py",
    "extsensorsim/ex4a_basicerrs_scal2d.py",
    "extsensorsim/ex4b_fielderrs_scal3d.py",
    "extsensorsim/ex4c_angleerrs_vec2d.py",
    "extsensorsim/ex4d_fieldlockerrs_vec3d.py",
    "extsensorsim/ex4e_chainfielderrs_vec2d.py",
    "extsensorsim/ex4f_caliberrs_scal2d.py",
    "extsensorsim/ex4g_spatavgerrs_scal2d.py",
    "extsensorsim/ex5a_expsim_thermmech2d.py",
    "extsensorsim/ex5b_expsim_thermmech3d.py",
)


@pytest.mark.example
@pytest.mark.parametrize("example", _BASIC_EXAMPLES)
def test_basic_sensorsim_example(
    run_example: Callable[..., Path],
    example: str,
) -> None:
    """Each basic Sensorsim gallery example runs without a GUI."""
    run_example(example, ("pyvale-output",), timeout=300.0)


@pytest.mark.example
@pytest.mark.example_slow
@pytest.mark.parametrize("example", _EXTENDED_EXAMPLES)
def test_extended_sensorsim_example(
    run_example: Callable[..., Path],
    example: str,
) -> None:
    """Each extended Sensorsim gallery example runs without a GUI."""
    expected_outputs = ()
    if not example.endswith("ex2_meshfreesensors.py"):
        expected_outputs = ("pyvale-output",)

    run_example(example, expected_outputs, timeout=300.0)
