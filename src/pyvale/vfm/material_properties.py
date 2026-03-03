from dataclasses import dataclass

import numpy.typing as npt
import numpy as np

@dataclass(slots=True)
class MaterialProperties:
    youngs_modulus: float
    poissons_ratio: float
    yield_strength: npt.NDArray[np.float64]
    hardening_modulus: npt.NDArray[np.float64]

