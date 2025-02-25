import math

class Interval:
    def __init__(self, min: float = None, max: float = None, *, a: "Interval" = None, b: "Interval" = None) -> None:
        self.min: float = math.inf
        self.max: float = -math.inf

        if min is not None:
            self.min = min
            self.max = max
        elif a is not None:
            # Create the interval tightly enclosing the two input intervals
            self.min = a.min if a.min < b.min else b.min
            self.max = a.max if a.max > b.max else b.max
    
    def size(self) -> float:
        return self.max - self.min
    
    def contains(self, x: float) -> bool:
        return self.min <= x and x <= self.max
    
    def surrounds(self, x: float) -> bool:
        return self.min < x and x < self.max
    
    def expand(self, delta: float) -> "Interval":
        padding = delta / 2
        return Interval(self.min - padding, self.max + padding)