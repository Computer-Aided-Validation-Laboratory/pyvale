from abc import ABC, abstractmethod
import numpy as np

class ConstLaw(ABC):
    @abstractmethod
    def calcStress(params: dict[str,np.ndarray],strain: np.ndarray) -> np.ndarray:

class LinearElasticUniaxial(ConstLaw):
    def calcStress(params: dict[str,np.ndarray],
                   strain: np.ndarray) -> np.ndarray:
        stress = params["EMod"]*strain
        return stress
