from abc import abstractmethod
import numpy as np
from perlin import Perlin

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

class NoiseTexture(Texture):
    def __init__(self, scale: float):
        self.scale = scale
        self.noise = Perlin()
    
    def value(self, u: float, v: float, p: np.ndarray) -> np.ndarray:
        return np.array([.5, .5, .5]) * (1 + np.sin(self.scale * p[2] + 10 * self.noise.turb(p, 7)))
