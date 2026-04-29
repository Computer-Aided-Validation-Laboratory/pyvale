from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_law import ConstitutiveLaw
from pyvale.vfm.parameter import Parameter
from pyvale.vfm.radial_return import radial_return

@dataclass(slots=True)
class LinearHardening(ConstitutiveLaw):

  def calculate_stress(
      self,
      parameters: dict[str, Parameter],
      strain: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
        # radial_return(strain)
        ...
