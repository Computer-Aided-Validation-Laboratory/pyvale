from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class IHardeningFunction(ABC):
    @abstractmethod
    def hardening(
        self,
        constitutive_parameter_maps: dict[str, npt.NDArray[np.float64]],
        equivalent_plastic_strain: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Return current yield stress and its derivative for the active
        hardening law.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            yield_stress : Current yield stress at all datapoints
            delta_yield_stress_delta_equivalent_plastic_strain : Derivative of
                yield stress with respect to equivalent plastic strain
                (i.e. hardening slope)
        """
        pass
