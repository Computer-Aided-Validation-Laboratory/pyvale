"""Real, portable Rabbit/Gmsh integration fixtures; no local-install skips."""

from pathlib import Path

import pytest

import pyvale.data as dataset


@pytest.fixture
def input_tree() -> Path:
    return dataset.mooseherder_data_dir()
