import math

class Interval:
    min: float
    max: float

    def __init__(self, min: None, max: None) -> None:
        if min == None and max == None:
            self.min = math.inf
            self.max = -math.inf
        else:
            self.min = min
            self.max = max
    
    def size(self) -> float:
        return self.max - self.min
    
    def contains(self, x: float) -> bool:
        return self.min <= x and x <= self.max
    
    def surrounds(self, x: float) -> bool:
        return self.min < x and x < self.max
    
    