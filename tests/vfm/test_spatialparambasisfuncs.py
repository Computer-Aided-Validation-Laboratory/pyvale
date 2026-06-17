import numpy as np
import pytest

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.spatialparambasisfuncs import BasisFunctionSpatialParameterisation


def _make_four_bump_map(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Generate a target map with four Gaussian bumps, one in each corner."""
    sigma = 0.1
    corners = [
        (0.2, 0.2),  # bottom-left
        (0.8, 0.2),  # bottom-right
        (0.2, 0.8),  # top-left
        (0.8, 0.8),  # top-right
    ]
    return sum(
        np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))
        for x0, y0 in corners
    )


# @pytest.fixture
def grid() -> tuple[np.ndarray, np.ndarray]:
    """Create a uniform 2D coordinate grid over [0, 1] x [0, 1]."""
    n_pts = 50
    x_1d = np.linspace(0.0, 1.0, n_pts)
    y_1d = np.linspace(0.0, 1.0, n_pts)
    x, y = np.meshgrid(x_1d, y_1d)
    return x, y


def test_basis_function_fits_four_gaussian_bumps(
    grid: tuple[np.ndarray, np.ndarray],
) -> None:
    """Regression test: BasisFunctionSpatialParameterisation should fit a target map
    composed of four Gaussian bumps, one in each corner, within a reasonable tolerance."""
    x, y = grid
    target_map = _make_four_bump_map(x, y)

    # Each bump peaks at ~1.0 and the bumps barely overlap, so the maximum
    # is ~1.0. Bounds span the full range of values.
    constitutive_parameter = ConstitutiveParameter(
        value=target_map,
        lower_bound=0.0,
        upper_bound=4.0,
    )

    parameterisation = BasisFunctionSpatialParameterisation(x, y)
    parameterisation.update_from_constitutive_parameter(constitutive_parameter)

    fitted_map = parameterisation.to_map(np.array(target_map.shape))

    # Assert the fitted map approximates the target within 10% normalised RMS error.
    # Normalised by the peak value to avoid division issues near zero.
    rmspe = np.sqrt(np.mean(((target_map - fitted_map) / target_map.max()) ** 2))
    assert rmspe < 0.1, (
        f"Fitted map RMSPE ({rmspe:.4f}) exceeded tolerance of 0.1. "
        "The basis function parameterisation failed to adequately fit four Gaussian bumps."
    )

test_basis_function_fits_four_gaussian_bumps(grid())
