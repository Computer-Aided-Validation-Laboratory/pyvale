from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from plots import plot_map_comparison
from rms import rms

from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.hardening import HardeningLinear
from pyvale.vfm.identification import run_identification
from pyvale.vfm.identificationconfig import (
    IdentificationConfig,
    IdentificationPhase,
)
from pyvale.vfm.metricsbvf import MetricSBVF
from pyvale.vfm.objectivefuncvector import VectorFirstResultPassthrough
from pyvale.vfm.optimiserleastsquares import OptimiserLeastSquares
from pyvale.vfm.spatialparambasisfuncs import (
    BasisFunctionKernelBivariate,
    BasisFunctionKernelBivariateSPD,
    BasisFunctionKernelUnivariate,
    SpatialParameterisationBasisFunction,
)
from pyvale.vfm.spatialparamknown import SpatialParameterisationKnown

EXPERIMENT_DATA_FILE = (
    Path(__file__).parent
    / "input"
    / "hole2d_plas_het"
    / "experiment_data.yaml"
)

KNOWN_PARAMETERS_FILE = (
    Path(__file__).parent
    / "gold"
    / "hole2d_plas_het.npz"
)

PLOT_INITIALISATION = False

PLOT_IDENTIFICATION = False


def test_basis_parameterisation_accepts_unshared_kernels() -> None:
    x, y = _unit_grid(2)
    parameterisation = SpatialParameterisationBasisFunction(
        x=x,
        y=y,
        kernels=[BasisFunctionKernelUnivariate(0.0, 0.0, 1.0)],
        heights=[2.0],
    )

    assert parameterisation.support.kernels is parameterisation.kernels
    np.testing.assert_allclose(
        parameterisation.to_map(np.asarray(x.shape, dtype=np.uint32)),
        2.0 * np.exp(-0.5 * (x**2 + y**2)),
    )


def _make_diagonal_bump_map(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    num_diagonal_bumps: int,
    diagonal_margin: float,
    bump_height: float,
    bump_sigma: float
) -> npt.NDArray[np.float64]:
    """Generate a target map with Gaussian bumps along the main diagonal.

    ``num_diagonal_bumps`` bumps are placed at evenly spaced points running from
    one corner of the grid diagonally to the opposite corner (bottom-left to
    top-right).
    """
    centres = np.linspace(
        diagonal_margin, 1.0 - diagonal_margin, num_diagonal_bumps
    )
    return sum(
        bump_height
        * np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2.0 * bump_sigma ** 2))
        for c in centres
    )


def _unit_grid(
    grid_points: int
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
]:
    """Create a uniform 2D coordinate grid over [0, 1] x [0, 1]."""
    x_1d = np.linspace(0.0, 1.0, grid_points)
    y_1d = np.linspace(0.0, 1.0, grid_points)
    x, y = np.meshgrid(x_1d, y_1d)
    return x, y


def test_initialisation_fits_diagonal_gaussian_bumps() -> None:
    """A basis function parameterisation initialised from a diagonal line of
    bumps should place one kernel per bump and reproduce the map to within
    tolerance."""

    grid_points = 101
    bump_sigma = 0.08
    bump_height = 1.0
    num_diagonal_bumps = 5
    diagonal_margin = 0.15  # keep the end bumps inside the domain

    # Fit-quality tolerances for the initialisation. The bumps are isotropic
    # Gaussians and the parameterisation resolves one univariate kernel per bump,
    # so the target is reproduced to essentially machine precision. These tight
    # tolerances keep a safe margin over that while still asserting a good fit.
    init_max_abs_diff_tolerance = 0.01 * bump_height  # 1% of the peak
    init_abs_perc_diff_tolerance = 1.0  # %, evaluated where the target is large
    significance_fraction = 0.1  # fraction of the peak above which a point counts


    x, y = _unit_grid(grid_points)

    target_map = _make_diagonal_bump_map(
        x,
        y,
        num_diagonal_bumps,
        diagonal_margin,
        bump_height,
        bump_sigma
    )

    # Each bump peaks at BUMP_HEIGHT and the bumps barely overlap. Bounds span
    # the full range of possible summed values.
    constitutive_parameter = ConstitutiveParameter(
        value=target_map,
        lower_bound=0.0,
        upper_bound=4.0 * bump_height,
    )

    parameterisation = SpatialParameterisationBasisFunction(x, y)
    parameterisation.fit_to_parameter_map(
        constitutive_parameter
    )

    # One kernel should have been placed for each bump.
    assert len(parameterisation.kernels) == num_diagonal_bumps, (
        f"expected {num_diagonal_bumps} kernels (one per bump), got "
        f"{len(parameterisation.kernels)}"
    )

    fitted_map = parameterisation.to_map(np.array(target_map.shape))

    abs_diff = np.abs(fitted_map - target_map)

    # Percentage difference is only meaningful where the target is
    # appreciably non-zero (away from the Gaussian tails).
    significant = target_map > significance_fraction * target_map.max()
    abs_perc_diff = np.abs(fitted_map - target_map) / np.abs(target_map) * 100.0
    mean_abs_perc_diff = float(np.mean(abs_perc_diff[significant]))

    print(f"initialisation: num kernels = {len(parameterisation.kernels)}")
    print(f"initialisation: max abs diff = {abs_diff.max():.6f}")
    print(
        f"initialisation: mean abs % diff (significant region) = "
        f"{mean_abs_perc_diff:.4f} %"
    )

    if PLOT_INITIALISATION:
        plot_map_comparison(
            x, y, target_map, fitted_map, "target", "fitted",
        )

    # The fitted map should closely reproduce the target map.
    assert abs_diff.max() < init_max_abs_diff_tolerance
    assert mean_abs_perc_diff < init_abs_perc_diff_tolerance


@pytest.mark.skip(reason="needs revised")
def test_identification_of_heterogeneous_yield_strength() -> None:
    """Identify a heterogeneous yield strength field with a basis function
    parameterisation while all other parameters are known, then compare the
    identified field against the known analytic field."""

    # Tolerances on the identified heterogeneous yield strength field. These are
    # provisional starting points that have not yet been calibrated against a full
    # converged run; tune them once test 2 has been run end-to-end. A homogeneous
    # initial guess now yields ~4 kernels (16 DOFs), below the SBVF residual count,
    # so the least-squares solve is well posed.
    ident_abs_diff_rms_tolerance = 20.0  # mpA
    ident_abs_perc_diff_tolerance = 10.0  # %

    experiment_data = ExperimentData.load_from_file(EXPERIMENT_DATA_FILE)

    known_parameter_maps = dict(np.load(KNOWN_PARAMETERS_FILE))

    x = experiment_data.specimen_geometry.x
    y = experiment_data.specimen_geometry.y

    constitutive_law = IsotropicVonMisesElastoplasticity(HardeningLinear())

    parameter_map_size = np.array(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32
    )

    # Known parameters are supplied at their true values; the yield strength is
    # given a homogeneous initial guess and identified via basis functions.
    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            known_parameter_maps["elastic_modulus"], 100_000.0, 500_000.0
        ),
        "poissons_ratio": ConstitutiveParameter(
            known_parameter_maps["poissons_ratio"], 0.1, 0.5
        ),
        "yield_strength": ConstitutiveParameter(
            np.max(known_parameter_maps["yield_strength"]),
            150.0,
            400.0,
            parameter_map_size
        ),
        "hardening_modulus": ConstitutiveParameter(
            known_parameter_maps["hardening_modulus"], 500.0, 10_000.0
        ),
    }

    metric = MetricSBVF(np.array([15, 15]))

    phases = [
        IdentificationPhase(
            {
                "elastic_modulus": [SpatialParameterisationKnown()],
                "poissons_ratio": [SpatialParameterisationKnown()],
                "yield_strength": [
                    SpatialParameterisationBasisFunction(x, y)
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
    # known_yield_map = _known_yield_map(x, y)

    roi = experiment_data.specimen_geometry.region_of_interest
    specimen_mask = roi.sample_specimen_mask(x, y)

    # Only compare within the region of interest (exclude the hole and any
    # points outside the specimen).
    identified_yield_strength = np.where(
        specimen_mask,
        identified_yield_map,
        np.nan
    )

    known_yield_strength = np.where(
        specimen_mask,
        known_parameter_maps["yield_strength"],
        np.nan
    )

    abs_diff = np.abs(identified_yield_strength - known_yield_strength)
    abs_diff_rms = rms(abs_diff)
    abs_perc_diff = abs_diff / np.abs(known_yield_strength) * 100.0
    mean_abs_perc_diff = float(np.nanmean(abs_perc_diff))

    print(f"identification: abs diff rms = {abs_diff_rms:.6f} MPa")
    print(f"identification: mean abs % diff = {mean_abs_perc_diff:.4f} %")

    if PLOT_IDENTIFICATION:
        plot_map_comparison(
            x,
            y,
            known_yield_strength,
            identified_yield_strength,
            "known",
            "identified",
        )

    # The identified heterogeneous yield strength field should be close to the
    # known analytic field.
    assert abs_diff_rms < ident_abs_diff_rms_tolerance
    assert mean_abs_perc_diff < ident_abs_perc_diff_tolerance


def test_bivariate_kernel_canonical_values_remove_axis_and_period_ambiguity() -> None:
    first = BasisFunctionKernelBivariate(0.0, 0.0, 4.0, 1.0, 0.2)
    swapped = BasisFunctionKernelBivariate(0.0, 0.0, 1.0, 4.0, 0.2 - np.pi / 2)
    periodic = BasisFunctionKernelBivariate(0.0, 0.0, 4.0, 1.0, 0.2 + np.pi)

    assert first.canonical_values() == pytest.approx(swapped.canonical_values())
    assert first.canonical_values() == pytest.approx(periodic.canonical_values())


def test_spd_kernel_matches_equivalent_rotated_bivariate_kernel() -> None:
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, 21), np.linspace(-2.0, 2.0, 21))
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    covariance = rotation @ np.diag([0.8, 0.2]) @ rotation.T
    values, vectors = np.linalg.eigh(covariance)
    log_covariance = (vectors * np.log(values)) @ vectors.T
    spd = SpatialParameterisationBasisFunction(
        x,
        y,
        heights=[1.0],
        kernels=[
            BasisFunctionKernelBivariateSPD(
                0.0,
                0.0,
                log_covariance[0, 0],
                log_covariance[0, 1],
                log_covariance[1, 1],
                1.0,
            )
        ],
        kernel_type="bivariate_spd",
    )
    conventional = SpatialParameterisationBasisFunction(
        x,
        y,
        heights=[1.0],
        kernels=[BasisFunctionKernelBivariate(0.0, 0.0, 0.8, 0.2, angle)],
        kernel_type="bivariate",
    )

    assert np.allclose(spd.to_map(np.array(x.shape)), conventional.to_map(np.array(x.shape)))


def test_spd_kernel_creation_uses_linear_log_covariance_coordinates() -> None:
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    parameterisation = SpatialParameterisationBasisFunction(
        x, y, kernel_type="bivariate_spd"
    )
    kernel = parameterisation.create_kernel(
        0.5,
        0.5,
        DegreeOfFreedom(0.1, 0.01, 1.0, scaling="log"),
    )

    assert isinstance(kernel, BasisFunctionKernelBivariateSPD)
    assert np.linalg.eigvalsh(kernel.covariance()) == pytest.approx([0.1, 0.1])
    assert all(dof.scaling == "linear" for dof in kernel.collect_degrees_of_freedom())


def test_basis_initialisation_preserves_explicit_height_seed() -> None:
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    parameterisation = SpatialParameterisationBasisFunction(x, y)
    parameterisation.kernels.append(
        BasisFunctionKernelBivariate(0.5, 0.5, 0.1, 0.05, 0.0)
    )
    parameterisation.heights.append(DegreeOfFreedom(42.0, -100.0, 100.0))
    parameter = ConstitutiveParameter(np.full((3, 3), 300.0), 200.0, 700.0)

    parameterisation.initialise_from_constitutive_parameter(parameter)

    assert isinstance(parameterisation.heights[0], DegreeOfFreedom)
    assert parameterisation.heights[0].value == 42.0


def test_empty_basis_initialisation_does_not_create_structure() -> None:
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    parameter = ConstitutiveParameter(np.full((3, 3), 300.0), 200.0, 700.0)
    parameterisation = SpatialParameterisationBasisFunction(x, y)

    parameterisation.initialise_from_constitutive_parameter(parameter)

    assert parameterisation.kernels == []
    parameterisation.fit_to_parameter_map(parameter, max_basis_functions=1)
    assert len(parameterisation.kernels) == 1


def test_bivariate_basis_map_fitting_creates_only_bivariate_kernels() -> None:
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    parameter = ConstitutiveParameter(np.full((5, 5), 300.0), 200.0, 700.0)
    parameterisation = SpatialParameterisationBasisFunction(
        x,
        y,
        kernel_type="bivariate",
    )

    parameterisation.fit_to_parameter_map(parameter, max_basis_functions=1)

    assert len(parameterisation.kernels) == 1
    assert isinstance(
        parameterisation.kernels[0],
        BasisFunctionKernelBivariate,
    )


def test_basis_centre_bounds_can_span_twice_the_data_domain() -> None:
    x, y = np.meshgrid(np.linspace(10.0, 14.0, 5), np.linspace(-2.0, 2.0, 5))
    parameterisation = SpatialParameterisationBasisFunction(
        x,
        y,
        centre_bounds_span_factor=2.0,
    )
    parameterisation.fit_to_map(np.ones_like(x), parameter_range=4.0, max_basis_functions=1)

    kernel = parameterisation.kernels[0]
    assert isinstance(kernel.x, DegreeOfFreedom)
    assert isinstance(kernel.y, DegreeOfFreedom)
    assert kernel.x.lower_bound == pytest.approx(8.0)
    assert kernel.x.upper_bound == pytest.approx(16.0)
    assert kernel.y.lower_bound == pytest.approx(-4.0)
    assert kernel.y.upper_bound == pytest.approx(4.0)


def test_basis_map_fit_accepts_a_spatial_fit_mask() -> None:
    x, y = _unit_grid(9)
    target = np.where(x < 0.5, 1.0, 0.0)
    fit_mask = x < 0.5
    parameterisation = SpatialParameterisationBasisFunction(x, y)
    parameterisation.fit_to_map(
        target,
        parameter_range=4.0,
        max_basis_functions=1,
        fit_mask=fit_mask,
    )
    fitted = parameterisation.to_map(np.asarray(target.shape, dtype=np.uint32))
    assert np.isfinite(fitted[fit_mask]).all()


def test_rejected_map_fit_candidate_restores_all_accepted_dofs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = _unit_grid(11)
    parameterisation = SpatialParameterisationBasisFunction(x, y)
    target_map = np.ones_like(x)
    fitted_values: list[tuple[float, float]] = []

    def fake_fit(self, target_map, include_support_degrees_of_freedom):
        _ = target_map, include_support_degrees_of_freedom
        kernel = self.kernels[0]
        assert isinstance(kernel.x, DegreeOfFreedom)
        assert isinstance(self.heights[0], DegreeOfFreedom)
        if not fitted_values:
            kernel.x.value = 0.25
            self.heights[0].value = 0.5
            fitted_values.append((kernel.x.value, self.heights[0].value))
            return
        kernel.x.value = 0.9
        self.heights[0].value = 3.0

    monkeypatch.setattr(
        SpatialParameterisationBasisFunction,
        "_fit_internal_dofs",
        fake_fit,
    )
    parameterisation.fit_to_map(
        target_map,
        parameter_range=4.0,
        max_basis_functions=2,
        minimum_relative_improvement=2.0,
    )

    assert len(parameterisation.kernels) == 1
    kernel = parameterisation.kernels[0]
    assert isinstance(kernel.x, DegreeOfFreedom)
    assert isinstance(parameterisation.heights[0], DegreeOfFreedom)
    assert kernel.x.value == 0.25
    assert parameterisation.heights[0].value == 0.5
