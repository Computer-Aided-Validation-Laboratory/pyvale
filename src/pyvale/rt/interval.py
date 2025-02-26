import math
from dataclasses import dataclass

@dataclass
class Interval:
    min: float = math.inf
    max: float = -math.inf

    def __init__(self) -> None:
        self.min: float = math.inf
        self.max: float = -math.inf
      
    @classmethod
    def from_floats(cls, a: float, b: float) -> "Interval":
        cls = Interval()
        cls.min = min(a, b)
        cls.max = max(a, b)
        return cls

    @classmethod
    def from_intervals(cls, a: "Interval", b: "Interval") -> "Interval":
        # Create the interval tightly enclosing the two input intervals
        cls = Interval()
        cls.min = min(a.min, b.min)
        cls.max = max(a.max, b.max)
        return cls
    
    def size(self) -> float:
        return self.max - self.min
    
    def contains(self, x: float) -> bool:
        return self.min <= x and x <= self.max
    
    def surrounds(self, x: float) -> bool:
        return self.min < x and x < self.max
    
    def expand(self, delta: float) -> "Interval":
        padding = delta / 2
        return Interval.from_floats(self.min - padding, self.max + padding)