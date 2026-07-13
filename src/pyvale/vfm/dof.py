from dataclasses import dataclass


@dataclass(slots=True)
class DegreeOfFreedom:
    """
    A single scalar degree of freedom with bounds.

    Used by the optimiser to explore the design space. Values are typically
    normalised to ``[0, 1]`` during optimisation and denormalised to the
    physical range ``[lower_bound, upper_bound]`` for evaluation
    """

    value: float
    lower_bound: float
    upper_bound: float
