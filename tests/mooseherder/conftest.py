# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Real, portable Rabbit/Gmsh integration fixtures; no local-install skips."""

from pathlib import Path

import pytest


@pytest.fixture
def input_tree() -> Path:
    return Path(__file__).resolve().parent / "fixtures"
