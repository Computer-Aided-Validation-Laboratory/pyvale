import numpy as np

class Ray:
    origin: np.ndarray
    direction: np.ndarray

    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = direction

    def at(self, t):
        return self.origin + t*self.direction