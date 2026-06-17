from dataclasses import dataclass


@dataclass(slots=True)
class DegreeOfFreedom:
    value: float
    lower_bound: float
    upper_bound: float
