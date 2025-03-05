from abc import abstractmethod
from dataclasses import dataclass
from typing import Text, Tuple
import numpy as np
from texture import Texture, SolidColor
from ray import Ray

class Material:
    @abstractmethod
    def scatter(self, r_in: Ray, rec) -> Tuple[np.ndarray, np.ndarray]:
        return None
    
    @abstractmethod
    def emitted(self, u: float, v: float, p: np.ndarray) -> np.ndarray:
        return np.array([0,0,0])

@dataclass
class Lambertian(Material):
    def __init__(self, texture: Texture) -> None:
        super().__init__()
        self.tex: Texture = texture
    
    @classmethod
    def from_colour(cls, albedo: np.ndarray) -> "Lambertian":
        solid_tex = SolidColor(albedo)
        return cls(solid_tex)
    
    def scatter(self, r_in: Ray, rec) -> Tuple[np.ndarray, np.ndarray]:
        scatter_direction = rec.normal + random_unit_vector()

        # Catch degenerate scatter direction
        # Helps accuracy. Very slow
        if np.allclose(scatter_direction, 0):
            scatter_direction = rec.normal
        
        scattered = Ray(rec.p, scatter_direction)
        attenuation = self.tex.value(rec.u, rec.v, rec.p)
        return (attenuation, scattered)

# class Metal(Material):
#     albedo: np.ndarray
#     fuzz: float

#     def __init__(self, color, fuzz: float) -> None:
#         self.albedo = color
#         self.fuzz = fuzz if fuzz < 1 else 1
    
#     def scatter(self, r_in: Ray, rec) -> Tuple[np.ndarray, np.ndarray]:
#         reflected = reflect
        
class DiffuseLight(Material):
    def __init__(self, tex: Texture) -> None:
        super().__init__()
        self.tex: Texture = tex
    
    @classmethod
    def from_colour(cls, emit: np.ndarray) -> "DiffuseLight":
        return cls(SolidColor(emit))
    
    def emitted(self, u: float, v: float, p: np.ndarray) -> np.ndarray:
        return self.tex.value(u, v, p)


def random_on_hemisphere(normal: np.ndarray):
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    if np.dot(vec, normal) > 0.0: # same hemesphere
        return vec
    else:
        return -vec

def random_unit_vector():
    v = np.random.randn(3)
    return v / np.linalg.norm(v)