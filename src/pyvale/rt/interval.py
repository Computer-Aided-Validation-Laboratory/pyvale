import math

class Interval:
    min: float
    max: float

    def __init__(self, min: float = None, max: float = None, *, a: "Interval" = None, b: "Interval" = None) -> None:
        if min == None and max == None:
            self.min = math.inf
            self.max = -math.inf
        elif min:
            self.min = min
            self.max = max
        elif a:
            # Create the interval tightly enclosing the two input intervals
            self.min = a.min if a.min < b.min else b.min
            self.max = a.max if a.max < b.max else b.max
        else:
            # shouldnt hit here
            assert(False)
    
    def size(self) -> float:
        return self.max - self.min
    
    def contains(self, x: float) -> bool:
        return self.min <= x and x <= self.max
    
    def surrounds(self, x: float) -> bool:
        return self.min < x and x < self.max
    
    def expand(self, delta: float) -> "Interval":
        padding = delta / 2
        return Interval(self.min - padding, self.max + padding)