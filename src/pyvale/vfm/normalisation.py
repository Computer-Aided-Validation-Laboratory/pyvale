from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

from pyvale.vfm.dof import DegreeOfFreedom


NormalisationScaling = Literal["linear", "log"]


def normalise_degree_of_freedom(
    degree_of_freedom: DegreeOfFreedom,
    scaling: NormalisationScaling = "linear",
) -> float:
    """
    Normalise a single degree of freedom to ``[0, 1]``.

    Parameters
    ----------
    degree_of_freedom : DegreeOfFreedom
        DOF with ``value``, ``lower_bound``, and ``upper_bound``

    Returns
    -------
    float
        Normalised value in ``[0, 1]``
    """
    return normalise_value(
        degree_of_freedom.value,
        degree_of_freedom.lower_bound,
        degree_of_freedom.upper_bound,
        scaling=scaling,
    )


def normalise_value(
    value: float,
    lower_bound: float,
    upper_bound: float,
    *,
    scaling: NormalisationScaling = "linear",
) -> float:
    """Normalise a scalar using linear or logarithmic bound scaling."""

    _validate_bounds(lower_bound, upper_bound)
    if scaling == "linear":
        return (
            (value - lower_bound)
            / (upper_bound - lower_bound)
        )
    if scaling == "log":
        _validate_log_inputs(value, lower_bound, upper_bound)
        return (
            (np.log(value) - np.log(lower_bound))
            / (np.log(upper_bound) - np.log(lower_bound))
        )
    raise ValueError(f"Unsupported normalisation scaling '{scaling}'.")


def _validate_bounds(
    lower_bound: float,
    upper_bound: float,
) -> None:
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        raise ValueError("Normalisation bounds must be finite.")
    if lower_bound >= upper_bound:
        raise ValueError(
            f"lower_bound ({lower_bound}) must be less than upper_bound ({upper_bound})."
        )


def _validate_log_inputs(
    value: float | npt.NDArray[np.float64],
    lower_bound: float | npt.NDArray[np.float64],
    upper_bound: float | npt.NDArray[np.float64],
) -> None:
    if np.any(np.asarray(value) <= 0.0):
        raise ValueError("Log normalisation requires positive values.")
    if np.any(np.asarray(lower_bound) <= 0.0):
        raise ValueError("Log normalisation requires positive lower bounds.")
    if np.any(np.asarray(upper_bound) <= 0.0):
        raise ValueError("Log normalisation requires positive upper bounds.")


def _resolve_scalings(
    scaling: NormalisationScaling | Sequence[NormalisationScaling],
    size: int,
) -> list[NormalisationScaling]:
    if isinstance(scaling, str):
        return [scaling] * size
    if len(scaling) != size:
        raise ValueError(
            f"Expected {size} scaling entries, got {len(scaling)}."
        )
    return list(scaling)


def _as_float_array(
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def _validate_array_shapes(
    normalised_values: npt.NDArray[np.float64],
    lower_bounds: npt.NDArray[np.float64],
    upper_bounds: npt.NDArray[np.float64],
) -> None:
    if (
        normalised_values.shape != lower_bounds.shape
        or normalised_values.shape != upper_bounds.shape
    ):
        raise ValueError(
            "normalised_values, lower_bounds, and upper_bounds must have matching shapes."
        )


def _validate_bound_arrays(
    lower_bounds: npt.NDArray[np.float64],
    upper_bounds: npt.NDArray[np.float64],
) -> None:
    if np.any(~np.isfinite(lower_bounds)) or np.any(~np.isfinite(upper_bounds)):
        raise ValueError("Normalisation bounds must be finite.")
    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Each lower bound must be less than its upper bound.")


def normalise_degrees_of_freedom(
    degrees_of_freedom: list[DegreeOfFreedom],
    scaling: NormalisationScaling | Sequence[NormalisationScaling] = "linear",
) -> npt.NDArray[np.float64]:
    """
    Normalise a list of degrees of freedom to ``[0, 1]``.

    Parameters
    ----------
    degrees_of_freedom : list[DegreeOfFreedom]
        One or more DOFs to normalise

    Returns
    -------
    npt.NDArray[np.float64]
        1D array of normalised values
    """
    scalings = _resolve_scalings(scaling, len(degrees_of_freedom))
    return np.asarray(
        [
            normalise_degree_of_freedom(dof, scaling=current_scaling)
            for dof, current_scaling in zip(
                degrees_of_freedom,
                scalings,
                strict=True,
            )
        ],
        dtype=np.float64,
    )


def denormalise_degree_of_freedom(
    normalised_value: float,
    lower_bound: float,
    upper_bound: float,
    scaling: NormalisationScaling = "linear",
) -> float:
    """
    Reverse the normalisation from ``[0, 1]`` back to physical units.

    Parameters
    ----------
    normalised_value : float
        Value in ``[0, 1]``
    lower_bound : float
        Physical lower bound
    upper_bound : float
        Physical upper bound

    Returns
    -------
    float
        Denormalised value in ``[lower_bound, upper_bound]``.
    """
    _validate_bounds(lower_bound, upper_bound)
    if scaling == "linear":
        return ((upper_bound - lower_bound) * normalised_value) + lower_bound
    if scaling == "log":
        _validate_log_inputs(1.0, lower_bound, upper_bound)
        return float(
            np.exp(
                np.log(lower_bound)
                + normalised_value
                * (np.log(upper_bound) - np.log(lower_bound))
            )
        )
    raise ValueError(f"Unsupported normalisation scaling '{scaling}'.")


def denormalise_degrees_of_freedom(
    normalised_values: npt.NDArray[np.float64],
    lower_bounds: npt.NDArray[np.float64],
    upper_bounds: npt.NDArray[np.float64],
    scaling: NormalisationScaling | Sequence[NormalisationScaling] = "linear",
) -> npt.NDArray[np.float64]:
    """
    Reverse the normalisation for an array of values.

    Parameters
    ----------
    normalised_values : npt.NDArray[np.float64]
        Values in ``[0, 1]``
    lower_bounds : npt.NDArray[np.float64]
        Physical lower bounds per DOF
    upper_bounds : npt.NDArray[np.float64]
        Physical upper bounds per DOF

    Returns
    -------
    npt.NDArray[np.float64]
        1D array of denormalised values
    """
    resolved_normalised_values = _as_float_array(normalised_values)
    resolved_lower_bounds = _as_float_array(lower_bounds)
    resolved_upper_bounds = _as_float_array(upper_bounds)
    _validate_array_shapes(
        resolved_normalised_values,
        resolved_lower_bounds,
        resolved_upper_bounds,
    )
    _validate_bound_arrays(resolved_lower_bounds, resolved_upper_bounds)

    scalings = _resolve_scalings(
        scaling,
        resolved_normalised_values.size,
    )
    if all(current_scaling == "linear" for current_scaling in scalings):
        return (
            (
                (resolved_upper_bounds - resolved_lower_bounds)
                * resolved_normalised_values
            )
            + resolved_lower_bounds
        )

    return np.asarray(
        [
            denormalise_degree_of_freedom(
                float(normalised_value),
                float(lower_bound),
                float(upper_bound),
                scaling=current_scaling,
            )
            for normalised_value, lower_bound, upper_bound, current_scaling in zip(
                resolved_normalised_values,
                resolved_lower_bounds,
                resolved_upper_bounds,
                scalings,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
