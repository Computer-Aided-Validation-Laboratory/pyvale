from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from pyvale.vfm import (
    BoundaryConditions,
    ConstitutiveParameter,
    convert_mask_to_physical_roi,
    DegreeOfFreedom,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    EquilibriumGapMetric,
    ExperimentData,
    HardeningLinear,
    IConstitutiveLaw,
    IdentificationConfig,
    IdentificationPhase,
    IMetric,
    IObjectiveFunction,
    IOptimiser,
    IsotropicVonMisesElastoplasticity,
    ISpatialParameterisation,
    IVectorObjectiveFunction,
    MetricSBVF,
    run_identification,
    SliceConfig,
    SliceWiseForceReconstructionMetric,
    SpatialParameterisationBasisFunction,
    SpatialParameterisationHomogeneous,
    SpatialParameterisationKnown,
    SpecimenGeometry,
    SupportBasis,
    SupportSlice,
    VectorFirstResultPassthrough,
    VfmRegionOfInterest,
)
from pyvale.vfm.normalisation import (
    denormalise_degrees_of_freedom,
    normalise_degrees_of_freedom,
)
from pyvale.vfm.spatialparam import PhaseSpatialState
from pyvale.vfm.spatialparambasisfuncs import BasisFunctionKernelBivariate


INPUTS_PATH = Path(__file__).resolve().parent / "rob-data" / "notched-weld-input-data"
FE_DATA_PATH = Path(
    "/home/robh/1_Projects/vfmap-numerical-paper/data/"
    "notchedButtWeld_bilin_lin360420S_hom3700H_imDef_1.5/1-feData"
)
OUTPUT_PLOT_PATH = Path(__file__).resolve().parent / "call_vfm_bf_results.png"

KNOWN_ELASTIC_MODULUS = 190_000.0
KNOWN_POISSONS_RATIO = 0.28
INITIAL_YIELD_STRENGTH = 320.0
INITIAL_HARDENING_MODULUS = 3_000.0
INITIAL_BASIS_HEIGHT = 60.0
SBVF_MESH_SIZE = np.array([15, 15], dtype=np.uint32)
PHASE_1_MAX_OPTIMISER_EVALUATIONS = 12
PHASE_2_MAX_OPTIMISER_EVALUATIONS = 40
FORCE_ERROR_NUM_SLICES = 80
EGI_WINDOW_SIZE = np.array([29, 29], dtype=np.uint32)


class _BoundedLeastSquares(IOptimiser):
    """Caller-local bounded solve for basis-function DOFs.

    The library's generic optimiser currently uses unconstrained LM. For this
    example, keeping normalised basis-function DOFs inside [0, 1] prevents
    invalid kernel variances without changing the package implementation.
    """

    def __init__(
        self,
        label: str,
        max_nfev: int,
        dof_scaling: tuple[str, ...] | None = None,
    ) -> None:
        self.label = label
        self.max_nfev = max_nfev
        self.dof_scaling = dof_scaling

    def get_required_objective_function_type(self) -> type:
        return IVectorObjectiveFunction

    def optimise(
        self,
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: np.ndarray,
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        metrics: list[IMetric],
        objective_function: IObjectiveFunction,
        experiment_data: ExperimentData,
        progress_callback=None,
    ) -> dict[str, list[ISpatialParameterisation]]:
        _ = progress_callback
        phase_spatial_state = PhaseSpatialState(spatial_parameterisations)
        dofs = phase_spatial_state.collect_degrees_of_freedom()
        if len(dofs) == 0:
            return spatial_parameterisations

        scaling = self.dof_scaling or ("linear",) * len(dofs)
        lower_bounds = np.asarray([dof.lower_bound for dof in dofs], dtype=np.float64)
        upper_bounds = np.asarray([dof.upper_bound for dof in dofs], dtype=np.float64)
        normalised_dofs = normalise_degrees_of_freedom(dofs, scaling=scaling)

        result = least_squares(
            self._evaluate_candidate,
            np.clip(normalised_dofs, 0.0, 1.0),
            bounds=(
                np.zeros_like(normalised_dofs),
                np.ones_like(normalised_dofs),
            ),
            method="trf",
            max_nfev=self.max_nfev,
            args=(
                constitutive_law,
                parameter_map_size,
                phase_spatial_state,
                metrics,
                objective_function,
                experiment_data,
            ),
        )
        physical_dofs = denormalise_degrees_of_freedom(
            result.x,
            lower_bounds,
            upper_bounds,
            scaling=scaling,
        )
        print(f"{self.label} optimiser summary:")
        print(f"  nfev: {result.nfev}")
        print(f"  cost: {float(result.cost):.6e}")
        print(f"  normalised dofs: {result.x}")
        print(f"  physical dofs:   {physical_dofs}")

        optimised_phase_spatial_state = phase_spatial_state.copy()
        optimised_phase_spatial_state.update_from_degrees_of_freedom(physical_dofs)
        return optimised_phase_spatial_state.spatial_parameterisations

    def _evaluate_candidate(
        self,
        normalised_degrees_of_freedom: np.ndarray,
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: np.ndarray,
        phase_spatial_state: PhaseSpatialState,
        metrics: list[IMetric],
        objective_function: IObjectiveFunction,
        experiment_data: ExperimentData,
    ) -> np.ndarray:
        degrees_of_freedom = phase_spatial_state.collect_degrees_of_freedom()
        scaling = self.dof_scaling or ("linear",) * len(degrees_of_freedom)
        physical_degrees_of_freedom = denormalise_degrees_of_freedom(
            normalised_degrees_of_freedom,
            np.asarray(
                [dof.lower_bound for dof in degrees_of_freedom],
                dtype=np.float64,
            ),
            np.asarray(
                [dof.upper_bound for dof in degrees_of_freedom],
                dtype=np.float64,
            ),
            scaling=scaling,
        )

        updated_phase_spatial_state = phase_spatial_state.copy()
        updated_phase_spatial_state.update_from_degrees_of_freedom(
            physical_degrees_of_freedom,
        )
        parameter_maps = updated_phase_spatial_state.evaluate_parameter_maps(
            parameter_map_size
        )
        stress = constitutive_law.calculate_stress(
            experiment_data.strain,
            parameter_maps,
        )
        metric_results = [
            metric.evaluate(
                stress,
                constitutive_law,
                parameter_map_size,
                updated_phase_spatial_state.spatial_parameterisations,
                experiment_data,
            )
            for metric in metrics
        ]
        return np.asarray(objective_function.evaluate(metric_results), dtype=np.float64)


class _SeededBasisFunction(SpatialParameterisationBasisFunction):
    """Use the supplied bivariate seed instead of fitting it away in phase 2."""

    def initialise_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter,
    ) -> None:
        self._ensure_heights_match_support()


def _load_experiment_data() -> ExperimentData:
    x = np.load(INPUTS_PATH / "x.npy")
    y = np.load(INPUTS_PATH / "y.npy")
    specimen_mask = np.load(INPUTS_PATH / "specimen_mask.npy")
    roi_definition = convert_mask_to_physical_roi(
        specimen_mask,
        x=x,
        y=y,
        simplification_pixels=0.0,
    )
    specimen_geometry = SpecimenGeometry(
        x=x,
        y=y,
        region_of_interest=VfmRegionOfInterest.from_definition(roi_definition),
        thickness=1.8,
        # MetricSBVF multiplies area by 1e6 internally, so provide m^2 here.
        pixel_area=np.load(INPUTS_PATH / "pixel_area.npy") * 1e-6,
    )

    boundary_conditions = BoundaryConditions(
        EdgeConditions(
            min_x_edge=Edge(
                EEdgeCondition.Fixed,
                EEdgeCondition.Fixed,
            ),
            max_x_edge=Edge(
                EEdgeCondition.Traction,
                EEdgeCondition.Fixed,
            ),
            min_y_edge=Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Free,
            ),
            max_y_edge=Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Free,
            ),
        ),
        np.load(INPUTS_PATH / "force.npy"),
    )

    return ExperimentData(
        np.load(INPUTS_PATH / "strain.npy"),
        specimen_geometry,
        boundary_conditions,
        np.load(INPUTS_PATH / "time.npy"),
    )


def _make_yield_basis_support(
    x: np.ndarray,
    y: np.ndarray,
) -> SupportBasis:
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    x_span = x_max - x_min
    y_span = y_max - y_min
    domain_span = max(x_span, y_span)

    centre_x = 0.5 * (x_min + x_max)
    centre_y = 0.5 * (y_min + y_max)
    min_variance = 0.05**2
    max_variance = (0.80 * domain_span) ** 2

    return SupportBasis(
        x=x,
        y=y,
        kernels=[
            BasisFunctionKernelBivariate(
                x=DegreeOfFreedom(centre_x, x_min, x_max),
                y=DegreeOfFreedom(centre_y, y_min, y_max),
                variance_x=DegreeOfFreedom(
                    (0.35 * domain_span) ** 2,
                    min_variance,
                    max_variance,
                ),
                variance_y=DegreeOfFreedom(
                    2.0**2,
                    min_variance,
                    max_variance,
                ),
                angle=DegreeOfFreedom(
                    np.deg2rad(60.0),
                    -0.5 * np.pi,
                    0.5 * np.pi,
                ),
            )
        ],
    )


def _plot_identified_yield_strength(
    x: np.ndarray,
    y: np.ndarray,
    specimen_mask: np.ndarray,
    yield_strength: np.ndarray,
) -> None:
    yield_strength_to_plot = np.where(specimen_mask, yield_strength, np.nan)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = ax.imshow(
        yield_strength_to_plot,
        origin="lower",
        cmap="viridis",
        extent=(
            float(np.nanmin(x)),
            float(np.nanmax(x)),
            float(np.nanmin(y)),
            float(np.nanmax(y)),
        ),
        aspect="auto",
    )
    ax.set_title("Identified Yield Strength")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(image, ax=ax, label="MPa")
    fig.savefig(OUTPUT_PLOT_PATH, dpi=200)
    print(f"Saved basis-function identification plot to {OUTPUT_PLOT_PATH}")
    if plt.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()


def _make_parameters(
    parameter_map_size: np.ndarray,
) -> dict[str, ConstitutiveParameter]:
    return {
        "elastic_modulus": ConstitutiveParameter(
            KNOWN_ELASTIC_MODULUS,
            150_000,
            250_000,
            parameter_map_size,
        ),
        "poissons_ratio": ConstitutiveParameter(
            KNOWN_POISSONS_RATIO,
            0.2,
            0.4,
            parameter_map_size,
        ),
        "yield_strength": ConstitutiveParameter(
            INITIAL_YIELD_STRENGTH,
            100,
            1000,
            parameter_map_size,
        ),
        "hardening_modulus": ConstitutiveParameter(
            INITIAL_HARDENING_MODULUS,
            1000,
            10_000,
            parameter_map_size,
        ),
    }


def _make_phases(
    yield_basis_support: SupportBasis,
) -> list[IdentificationPhase]:
    return [
        IdentificationPhase(
            spatial_parameterisations={
                "elastic_modulus": [SpatialParameterisationKnown()],
                "poissons_ratio": [SpatialParameterisationKnown()],
                "yield_strength": [
                    SpatialParameterisationHomogeneous(),
                ],
                "hardening_modulus": [SpatialParameterisationHomogeneous()],
            },
            metrics=[
                MetricSBVF(
                    SBVF_MESH_SIZE,
                    vf_scaling_fraction=0.3,
                )
            ],
            objective_function=VectorFirstResultPassthrough(),
            optimiser=_BoundedLeastSquares(
                "phase 1 homogeneous",
                max_nfev=PHASE_1_MAX_OPTIMISER_EVALUATIONS,
            ),
        ),
        IdentificationPhase(
            spatial_parameterisations={
                "elastic_modulus": [SpatialParameterisationKnown()],
                "poissons_ratio": [SpatialParameterisationKnown()],
                "yield_strength": [
                    SpatialParameterisationHomogeneous(),
                    _SeededBasisFunction(
                        support=yield_basis_support,
                        heights=[
                            DegreeOfFreedom(
                                INITIAL_BASIS_HEIGHT,
                                -100.0,
                                100.0,
                            )
                        ],
                    ),
                ],
                "hardening_modulus": [SpatialParameterisationHomogeneous()],
            },
            metrics=[
                MetricSBVF(
                    SBVF_MESH_SIZE,
                    vf_scaling_fraction=0.3,
                )
            ],
            objective_function=VectorFirstResultPassthrough(),
            optimiser=_BoundedLeastSquares(
                "phase 2 homogeneous + bivariate BF",
                max_nfev=PHASE_2_MAX_OPTIMISER_EVALUATIONS,
                dof_scaling=(
                    "linear",
                    "linear",
                    "log",
                    "log",
                    "linear",
                    "linear",
                    "linear",
                    "linear",
                ),
            ),
        ),
    ]


def _copy_parameter_maps(
    parameter_maps: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        name: np.array(parameter_map, dtype=np.float64, copy=True)
        for name, parameter_map in parameter_maps.items()
    }


def _load_target_parameter_maps(
    x: np.ndarray,
    y: np.ndarray,
    specimen_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load FE material properties and sample them onto the DIC grid."""

    centroids = np.loadtxt(FE_DATA_PATH / "centroid_results.txt")
    material_assignments = np.loadtxt(FE_DATA_PATH / "matassarray_results.txt")
    material_properties = np.loadtxt(FE_DATA_PATH / "matproparray_results.txt")

    material_ids = material_assignments[:, 1].astype(np.int64)
    yield_strength_by_element = material_properties[material_ids - 1, 3]
    hardening_modulus_by_element = material_properties[material_ids - 1, 4]

    tree = cKDTree(centroids[:, 1:3])
    query_points = np.column_stack((x[specimen_mask], y[specimen_mask]))
    _, nearest_element_indices = tree.query(query_points)

    target_yield_strength = np.full(x.shape, np.nan, dtype=np.float64)
    target_hardening_modulus = np.full(x.shape, np.nan, dtype=np.float64)
    target_yield_strength[specimen_mask] = (
        yield_strength_by_element[nearest_element_indices]
    )
    target_hardening_modulus[specimen_mask] = (
        hardening_modulus_by_element[nearest_element_indices]
    )

    return {
        "elastic_modulus": np.full(x.shape, KNOWN_ELASTIC_MODULUS),
        "poissons_ratio": np.full(x.shape, KNOWN_POISSONS_RATIO),
        "yield_strength": target_yield_strength,
        "hardening_modulus": target_hardening_modulus,
    }


def _plot_map(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: TwoSlopeNorm | None = None,
) -> None:
    image = ax.imshow(
        values,
        origin="lower",
        cmap=cmap,
        extent=(
            float(np.nanmin(x)),
            float(np.nanmax(x)),
            float(np.nanmin(y)),
            float(np.nanmax(y)),
        ),
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.figure.colorbar(image, ax=ax, label=label)


def _symmetric_norm(values: np.ndarray) -> TwoSlopeNorm:
    max_abs = float(np.nanmax(np.abs(values)))
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        max_abs = 1.0
    return TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def _plot_identified_parameter_maps(
    *,
    phase_number: int,
    x: np.ndarray,
    y: np.ndarray,
    specimen_mask: np.ndarray,
    parameter_maps: dict[str, np.ndarray],
) -> None:
    yield_strength = np.where(
        specimen_mask,
        parameter_maps["yield_strength"],
        np.nan,
    )
    hardening_modulus = np.where(
        specimen_mask,
        parameter_maps["hardening_modulus"],
        np.nan,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _plot_map(
        axes[0],
        x,
        y,
        yield_strength,
        title=f"Phase {phase_number} Yield Strength",
        label="MPa",
        cmap="viridis",
        vmin=350.0,
        vmax=450.0,
    )
    _plot_map(
        axes[1],
        x,
        y,
        hardening_modulus,
        title=f"Phase {phase_number} Hardening Modulus",
        label="MPa",
        cmap="magma",
        vmin=3000.0,
        vmax=3800.0,
    )
    output_path = (
        Path(__file__).resolve().parent
        / f"call_vfm_bf_phase_{phase_number}_identified_maps.png"
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved phase {phase_number} identified maps to {output_path}")


def _plot_percent_error_maps(
    *,
    phase_number: int,
    x: np.ndarray,
    y: np.ndarray,
    specimen_mask: np.ndarray,
    parameter_maps: dict[str, np.ndarray],
    target_parameter_maps: dict[str, np.ndarray],
) -> None:
    yield_error = 100.0 * (
        parameter_maps["yield_strength"] - target_parameter_maps["yield_strength"]
    ) / target_parameter_maps["yield_strength"]
    hardening_error = 100.0 * (
        parameter_maps["hardening_modulus"] - target_parameter_maps["hardening_modulus"]
    ) / target_parameter_maps["hardening_modulus"]
    yield_error = np.where(specimen_mask, yield_error, np.nan)
    hardening_error = np.where(specimen_mask, hardening_error, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _plot_map(
        axes[0],
        x,
        y,
        yield_error,
        title=f"Phase {phase_number} Yield Strength Error",
        label="%",
        cmap="coolwarm",
        norm=_symmetric_norm(yield_error),
    )
    _plot_map(
        axes[1],
        x,
        y,
        hardening_error,
        title=f"Phase {phase_number} Hardening Modulus Error",
        label="%",
        cmap="coolwarm",
        norm=_symmetric_norm(hardening_error),
    )
    output_path = (
        Path(__file__).resolve().parent
        / f"call_vfm_bf_phase_{phase_number}_percent_error_maps.png"
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved phase {phase_number} percent error maps to {output_path}")


def _calculate_force_reconstruction_error_map(
    *,
    stress: np.ndarray,
    experiment_data: ExperimentData,
    force_reconstruction_metric: SliceWiseForceReconstructionMetric,
) -> tuple[np.ndarray, float]:
    """Paint the slice-wise force reconstruction metric back to the grid."""

    force_error = force_reconstruction_metric.evaluate_force_recon_error(
        stress,
        experiment_data,
    )
    slice_partition = force_reconstruction_metric.slice_partition
    if slice_partition is None:
        raise RuntimeError("Force reconstruction slice partition is not initialised.")

    force_error_map = np.full(
        experiment_data.specimen_geometry.x.shape,
        np.nan,
        dtype=np.float64,
    )
    slice_error_percent = 100.0 * force_error.weighted_temporal_rms
    for slice_index, slice_error in enumerate(slice_error_percent):
        force_error_map[slice_partition.get_slice_mask(slice_index)] = slice_error

    return force_error_map, 100.0 * force_error.weighted_spatiotemporal_rms


def _plot_force_reconstruction_error_map(
    *,
    phase_number: int,
    x: np.ndarray,
    y: np.ndarray,
    specimen_mask: np.ndarray,
    force_error_map: np.ndarray,
) -> None:
    force_error_map = np.where(specimen_mask, force_error_map, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    _plot_map(
        ax,
        x,
        y,
        force_error_map,
        title=f"Phase {phase_number} Fine-Slice Force Reconstruction Error",
        label="% peak applied force",
        cmap="inferno",
        vmin=0.0,
    )
    output_path = (
        Path(__file__).resolve().parent
        / f"call_vfm_bf_phase_{phase_number}_force_reconstruction_error.png"
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(
        f"Saved phase {phase_number} force reconstruction error map to {output_path}"
    )


def _plot_equilibrium_gap_map(
    *,
    phase_number: int,
    x: np.ndarray,
    y: np.ndarray,
    specimen_mask: np.ndarray,
    equilibrium_gap_map: np.ndarray,
) -> None:
    equilibrium_gap_map = np.where(specimen_mask, equilibrium_gap_map, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    _plot_map(
        ax,
        x,
        y,
        100.0 * equilibrium_gap_map,
        title=f"Phase {phase_number} Equilibrium Gap Indicator",
        label="normalised EGI (%)",
        cmap="inferno",
        vmin=0.0,
    )
    output_path = (
        Path(__file__).resolve().parent
        / f"call_vfm_bf_phase_{phase_number}_equilibrium_gap.png"
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved phase {phase_number} equilibrium gap map to {output_path}")


def _save_phase_diagnostics(
    *,
    phase_number: int,
    constitutive_law: IConstitutiveLaw,
    experiment_data: ExperimentData,
    parameter_maps: dict[str, np.ndarray],
    target_parameter_maps: dict[str, np.ndarray],
    force_reconstruction_metric: SliceWiseForceReconstructionMetric,
    equilibrium_gap_metric: EquilibriumGapMetric,
) -> None:
    specimen_geometry = experiment_data.specimen_geometry
    x = specimen_geometry.x
    y = specimen_geometry.y
    specimen_mask = specimen_geometry.specimen_mask

    _plot_identified_parameter_maps(
        phase_number=phase_number,
        x=x,
        y=y,
        specimen_mask=specimen_mask,
        parameter_maps=parameter_maps,
    )
    _plot_percent_error_maps(
        phase_number=phase_number,
        x=x,
        y=y,
        specimen_mask=specimen_mask,
        parameter_maps=parameter_maps,
        target_parameter_maps=target_parameter_maps,
    )

    stress = constitutive_law.calculate_stress(
        experiment_data.strain,
        parameter_maps,
    )
    (
        force_error_map,
        weighted_force_error_percent,
    ) = _calculate_force_reconstruction_error_map(
        stress=stress,
        experiment_data=experiment_data,
        force_reconstruction_metric=force_reconstruction_metric,
    )
    _plot_force_reconstruction_error_map(
        phase_number=phase_number,
        x=x,
        y=y,
        specimen_mask=specimen_mask,
        force_error_map=force_error_map,
    )
    print(
        f"Phase {phase_number} weighted slice force reconstruction error: "
        f"{weighted_force_error_percent:.6f}%"
    )

    equilibrium_gap = equilibrium_gap_metric.evaluate_equilibrium_gap(stress)
    _plot_equilibrium_gap_map(
        phase_number=phase_number,
        x=x,
        y=y,
        specimen_mask=specimen_mask,
        equilibrium_gap_map=equilibrium_gap.weighted_temporal_rms,
    )
    print(
        f"Phase {phase_number} weighted equilibrium gap indicator: "
        f"{100.0 * equilibrium_gap.weighted_spatiotemporal_rms:.6f}%"
    )


def main() -> None:
    experiment_data = _load_experiment_data()
    specimen_geometry = experiment_data.specimen_geometry
    parameter_map_size = np.array(specimen_geometry.x.shape, dtype=np.uint32)

    constitutive_law = IsotropicVonMisesElastoplasticity(HardeningLinear())
    yield_basis_support = _make_yield_basis_support(
        specimen_geometry.x,
        specimen_geometry.y,
    )

    parameters = _make_parameters(parameter_map_size)
    phases = _make_phases(yield_basis_support)
    target_parameter_maps = _load_target_parameter_maps(
        specimen_geometry.x,
        specimen_geometry.y,
        specimen_geometry.specimen_mask,
    )
    force_reconstruction_metric = SliceWiseForceReconstructionMetric(
        support=SupportSlice(
            slice_config=SliceConfig(
                axis="x",
                num_slices=FORCE_ERROR_NUM_SLICES,
            )
        )
    )
    force_reconstruction_metric.initialise(experiment_data)
    equilibrium_gap_metric = EquilibriumGapMetric(
        window_size=EGI_WINDOW_SIZE,
        # call_vfm_bf stores pixel area in m^2 for MetricSBVF compatibility.
        pixel_area_scale=1e6,
    )
    equilibrium_gap_metric.initialise(experiment_data)

    phase_parameter_maps: list[dict[str, np.ndarray]] = []
    for phase_index, phase in enumerate(phases, start=1):
        identification = IdentificationConfig(
            constitutive_law=constitutive_law,
            parameters=parameters,
            phases=[phase],
        )
        result = run_identification(experiment_data, identification)
        parameter_maps = _copy_parameter_maps(result.parameter_maps)
        phase_parameter_maps.append(parameter_maps)
        _save_phase_diagnostics(
            phase_number=phase_index,
            constitutive_law=constitutive_law,
            experiment_data=experiment_data,
            parameter_maps=parameter_maps,
            target_parameter_maps=target_parameter_maps,
            force_reconstruction_metric=force_reconstruction_metric,
            equilibrium_gap_metric=equilibrium_gap_metric,
        )

    final_parameter_maps = phase_parameter_maps[-1]
    yield_strength = final_parameter_maps["yield_strength"]
    yield_strength_roi = np.where(
        specimen_geometry.specimen_mask,
        yield_strength,
        np.nan,
    )

    print("Basis-function identification summary:")
    print(f"  yield_strength mean: {float(np.nanmean(yield_strength_roi)):.6f} MPa")
    print(f"  yield_strength min:  {float(np.nanmin(yield_strength_roi)):.6f} MPa")
    print(f"  yield_strength max:  {float(np.nanmax(yield_strength_roi)):.6f} MPa")
    print(
        "  hardening_modulus mean: "
        f"{float(np.nanmean(final_parameter_maps['hardening_modulus'])):.6f} MPa"
    )

    _plot_identified_yield_strength(
        specimen_geometry.x,
        specimen_geometry.y,
        specimen_geometry.specimen_mask,
        yield_strength,
    )


if __name__ == "__main__":
    main()
