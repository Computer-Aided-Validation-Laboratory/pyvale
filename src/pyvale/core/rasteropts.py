"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass


@dataclass(slots=True)
class RasterOpts:
    subsample: int = 2

    def __post_init__(self) -> None:
        assert self.subsample > 0, "Subsampling must be larger than 0."

