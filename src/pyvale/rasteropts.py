#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
from dataclasses import dataclass


@dataclass(slots=True)
class RasterOpts:
    background: float = 0.5
    bits: int = 16
    subsample: int = 2

    def __post_init__(self) -> None:
        assert self.subsample > 0, "Subsampling must be larger than 0."

