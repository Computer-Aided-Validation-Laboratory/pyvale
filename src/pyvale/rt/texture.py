from abc import abstractmethod
import numpy as np


class Texture:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def value(self, u: float, v: float, p: np.ndarray) -> np.ndarray:
        pass

class SolidColor(Texture):
    def __init__(self, albedo: np.ndarray) -> None:
        super().__init__()
        self.albedo = albedo
    
    def value(self, u: float, v: float, p: np.ndarray) -> np.ndarray:
        return self.albedo