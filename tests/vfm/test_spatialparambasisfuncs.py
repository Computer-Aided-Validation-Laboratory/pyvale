import numpy as np
import numpy.typing as npt
import pytest
from load_sim_data import (
    PLATE_WIDTH,
    load_force,
    load_strain,
    load_timesteps,
)
from plots import _plot_map_comparison
from utils import rms

from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import (
    BoundaryConditions,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.hardening import HardeningLinear
from pyvale.vfm.identification import run_identification
from pyvale.vfm.identificationconfig import (
    IdentificationConfig,
    IdentificationPhase,
)
from pyvale.vfm.metricsbvf import MetricSBVF
from pyvale.vfm.objectivefuncvector import VectorFirstResultPassthrough
from pyvale.vfm.optimiserleastsquares import OptimiserLeastSquares
from pyvale.vfm.roi import VfmRegionOfInterest, convert_mask_to_physical_roi
from pyvale.vfm.spatialparambasisfuncs import (
    SpatialParameterisationBasisFunction,
)
from pyvale.vfm.spatialparamknown import SpatialParameterisationKnown

# ---------------------------------------------------------------------------
# Test 1: initialisation of a basis function parameterisation from a map
# ---------------------------------------------------------------------------
GRID_POINTS = 101
BUMP_SIGMA = 0.08
BUMP_HEIGHT = 1.0
NUM_DIAGONAL_BUMPS = 5
DIAGONAL_MARGIN = 0.15  # keep the end bumps inside the domain

# Fit-quality tolerances for the initialisation. The bumps are isotropic
# Gaussians and the parameterisation resolves one univariate kernel per bump,
# so the target is reproduced to essentially machine precision. These tight
# tolerances keep a safe margin over that while still asserting a good fit.
INIT_MAX_ABS_DIFF_TOLERANCE = 0.01 * BUMP_HEIGHT  # 1% of the peak
INIT_ABS_PERC_DIFF_TOLERANCE = 1.0  # %, evaluated where the target is large
SIGNIFICANCE_FRACTION = 0.1  # fraction of the peak above which a point counts

PLOT_INITIALISATION = False

# ---------------------------------------------------------------------------
# Test 2: identification of a heterogeneous yield strength field
# ---------------------------------------------------------------------------
EXODUS_FILE_NAME = "out_hole2d_plas_het_32f.e"
GRID_DIVS = 101
PLATE_THICKNESS = 1e-3  # m

# Heterogeneous yield strength field baked into the FE model (see
# platehole2d_plas_het.i / common_het_geometry.i in the vfmverif data). A
# Gaussian bump sits at the geometric centre of the plate.
YIELD_INF = 200.0   # MPa, yield stress far from the bump
PEAK_YIELD = 240.0  # MPa, yield stress at the bump centre
YIELD_STD_X = PLATE_WIDTH / 2.0  # m
YIELD_STD_Y = PLATE_WIDTH / 4.0  # m

# Known (homogeneous) constitutive parameters. Only yield strength is
# heterogeneous and therefore identified.
KNOWN_ELASTIC_MODULUS = 200_000.0  # MPa
KNOWN_POISSONS_RATIO = 0.3
KNOWN_HARDENING_MODULUS = 1_000.0  # MPa

# Tolerances on the identified heterogeneous yield strength field. These are
# provisional starting points that have not yet been calibrated against a full
# converged run; tune them once test 2 has been run end-to-end. A homogeneous
# initial guess now yields ~4 kernels (16 DOFs), below the SBVF residual count,
# so the least-squares solve is well posed.
IDENT_ABS_DIFF_RMS_TOLERANCE = 20.0  # MPa
IDENT_ABS_PERC_DIFF_TOLERANCE = 10.0  # %

PLOT_IDENTIFICATION = False


def _make_diagonal_bump_map(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Generate a target map with Gaussian bumps along the main diagonal.

    ``NUM_DIAGONAL_BUMPS`` bumps are placed at evenly spaced points running from
    one corner of the grid diagonally to the opposite corner (bottom-left to
    top-right).
    """
    centres = np.linspace(
        DIAGONAL_MARGIN, 1.0 - DIAGONAL_MARGIN, NUM_DIAGONAL_BUMPS
    )
    return sum(
        BUMP_HEIGHT
        * np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2.0 * BUMP_SIGMA ** 2))
        for c in centres
    )


def _unit_grid() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Create a uniform 2D coordinate grid over [0, 1] x [0, 1]."""
    x_1d = np.linspace(0.0, 1.0, GRID_POINTS)
    y_1d = np.linspace(0.0, 1.0, GRID_POINTS)
    x, y = np.meshgrid(x_1d, y_1d)
    return x, y


def test_initialisation_fits_diagonal_gaussian_bumps() -> None:
    """A basis function parameterisation initialised from a diagonal line of
    bumps should place one kernel per bump and reproduce the map to within
    tolerance."""
    x, y = _unit_grid()
    target_map = _make_diagonal_bump_map(x, y)

    # Each bump peaks at BUMP_HEIGHT and the bumps barely overlap. Bounds span
    # the full range of possible summed values.
    constitutive_parameter = ConstitutiveParameter(
        value=target_map,
        lower_bound=0.0,
        upper_bound=4.0 * BUMP_HEIGHT,
    )

    parameterisation = SpatialParameterisationBasisFunction(x, y)
    parameterisation.initialise_from_constitutive_parameter(
        constitutive_parameter
    )

    # One kernel should have been placed for each bump.
    assert len(parameterisation.kernels) == NUM_DIAGONAL_BUMPS, (
        f"expected {NUM_DIAGONAL_BUMPS} kernels (one per bump), got "
        f"{len(parameterisation.kernels)}"
    )

    fitted_map = parameterisation.to_map(np.array(target_map.shape))

    abs_diff = np.abs(fitted_map - target_map)

    # Percentage difference is only meaningful where the target is
    # appreciably non-zero (away from the Gaussian tails).
    significant = target_map > SIGNIFICANCE_FRACTION * target_map.max()
    abs_perc_diff = np.abs(fitted_map - target_map) / np.abs(target_map) * 100.0
    mean_abs_perc_diff = float(np.mean(abs_perc_diff[significant]))

    print(f"initialisation: num kernels = {len(parameterisation.kernels)}")
    print(f"initialisation: max abs diff = {abs_diff.max():.6f}")
    print(
        f"initialisation: mean abs % diff (significant region) = "
        f"{mean_abs_perc_diff:.4f} %"
    )

    if PLOT_INITIALISATION:
        _plot_map_comparison(
            x, y, target_map, fitted_map, "target", "fitted",
        )

    # The fitted map should closely reproduce the target map.
    assert abs_diff.max() < INIT_MAX_ABS_DIFF_TOLERANCE
    assert mean_abs_perc_diff < INIT_ABS_PERC_DIFF_TOLERANCE


@pytest.mark.skip(reason="needs revised")
def test_identification_of_heterogeneous_yield_strength() -> None:
    """Identify a heterogeneous yield strength field with a basis function
    parameterisation while all other parameters are known, then compare the
    identified field against the known analytic field."""
    experiment_data = _setup_experiment_data()

    x_grid = experiment_data.specimen_geometry.x
    y_grid = experiment_data.specimen_geometry.y

    constitutive_law = IsotropicVonMisesElastoplasticity(HardeningLinear())

    parameter_map_size = np.array([GRID_DIVS, GRID_DIVS], dtype=np.uint32)

    # Known parameters are supplied at their true values; the yield strength is
    # given a homogeneous initial guess and identified via basis functions.
    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            KNOWN_ELASTIC_MODULUS, 100_000.0, 500_000.0, parameter_map_size
        ),
        "poissons_ratio": ConstitutiveParameter(
            KNOWN_POISSONS_RATIO, 0.1, 0.5, parameter_map_size
        ),
        "yield_strength": ConstitutiveParameter(
            YIELD_INF, 150.0, 400.0, parameter_map_size
        ),
        "hardening_modulus": ConstitutiveParameter(
            KNOWN_HARDENING_MODULUS, 500.0, 10_000.0, parameter_map_size
        ),
    }

    metric = MetricSBVF(np.array([15, 15]))

    phases = [
        IdentificationPhase(
            {
                "elastic_modulus": [SpatialParameterisationKnown()],
                "poissons_ratio": [SpatialParameterisationKnown()],
                "yield_strength": [
                    SpatialParameterisationBasisFunction(x_grid, y_grid)
                ],
                "hardening_modulus": [SpatialParameterisationKnown()],
            },
            [metric],
            VectorFirstResultPassthrough(),
            OptimiserLeastSquares(),
        )
    ]

    ident_config = IdentificationConfig(
        constitutive_law,
        parameters,
        phases,
    )

    print("Running identification...")
    result = run_identification(experiment_data, ident_config)

    identified_yield_map = result.parameter_maps["yield_strength"]
    known_yield_map = _known_yield_map(x_grid, y_grid)

    # Only compare within the region of interest (exclude the hole and any
    # points outside the specimen).
    roi = experiment_data.specimen_geometry.region_of_interest
    identified_roi = np.where(roi, identified_yield_map, np.nan)
    known_roi = np.where(roi, known_yield_map, np.nan)

    abs_diff = np.abs(identified_roi - known_roi)
    abs_diff_rms = rms(abs_diff)
    abs_perc_diff = abs_diff / np.abs(known_roi) * 100.0
    mean_abs_perc_diff = float(np.nanmean(abs_perc_diff))

    print(f"identification: abs diff rms = {abs_diff_rms:.6f} MPa")
    print(f"identification: mean abs % diff = {mean_abs_perc_diff:.4f} %")

    if PLOT_IDENTIFICATION:
        _plot_map_comparison(
            x_grid, y_grid, known_roi, identified_roi, "known", "identified",
        )

    # The identified heterogeneous yield strength field should be close to the
    # known analytic field.
    assert abs_diff_rms < IDENT_ABS_DIFF_RMS_TOLERANCE
    assert mean_abs_perc_diff < IDENT_ABS_PERC_DIFF_TOLERANCE


def _known_yield_map(
    x_grid: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Evaluate the known heterogeneous yield strength field on the grid.

    Mirrors the FE model definition (a Gaussian bump at the geometric centre of
    the plate) in the pyvale coordinate frame:

        yield = YieldInf + (PeakYield - YieldInf)
                * exp(-0.5 * (((x - cx) / stdX)^2 + ((y - cy) / stdY)^2))
    """
    centre_x = 0.5 * (np.nanmin(x_grid) + np.nanmax(x_grid))
    centre_y = 0.5 * (np.nanmin(y_grid) + np.nanmax(y_grid))

    return YIELD_INF + (PEAK_YIELD - YIELD_INF) * np.exp(
        -0.5
        * (
            ((x_grid - centre_x) / YIELD_STD_X) ** 2
            + ((y_grid - centre_y) / YIELD_STD_Y) ** 2
        )
    )


def _setup_experiment_data() -> ExperimentData:
    (x_grid, y_grid, strain) = load_strain(EXODUS_FILE_NAME, GRID_DIVS)
    force = load_force(EXODUS_FILE_NAME)
    timesteps = load_timesteps(EXODUS_FILE_NAME)

    specimen_mask = ~np.isnan(strain[0, 0, :, :])

    grid_element_area = (
        (x_grid[0, 1] - x_grid[0, 0]) * (y_grid[1, 0] - y_grid[0, 0])
    )

    roi = VfmRegionOfInterest.from_definition(
        convert_mask_to_physical_roi(
            specimen_mask,
            x_grid,
            y_grid,
            simplification_pixels=0.0
        )
    )

    specimen_geometry = SpecimenGeometry(
        x_grid,
        y_grid,
        np.full_like(x_grid, grid_element_area, dtype=np.float64),
        PLATE_THICKNESS,
        roi
    )

    # seems to be an issue with FE input force data being 1000x too large
    force *= 1e-3

    boundary_conditions = BoundaryConditions(
        EdgeConditions(
            min_x_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Free),
            max_x_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Free),
            min_y_edge=Edge(x=EEdgeCondition.Fixed, y=EEdgeCondition.Fixed),
            max_y_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Traction),
        ),
        force,
    )

    return ExperimentData(
        strain,
        specimen_geometry,
        boundary_conditions,
        timesteps,
    )
