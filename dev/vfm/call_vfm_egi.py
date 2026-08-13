from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyvale.vfm import (
    BoundaryConditions,
    ConstitutiveParameter,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    HardeningLinear,
    IdentificationConfig,
    IdentificationPhase,
    IsotropicVonMisesElastoplasticity,
    run_identification,
    SliceConfig,
    SliceMergeSplitRefinement,
    SliceWiseForceReconstructionMetric,
    SliceWiseIndependentLeastSquares,
    SliceWiseSpatialParameterisation,
    SpatialParameterisationKnown,
    SpatialParameterisationHomogeneous,
    SpecimenGeometry,
    SupportSlice,
    VectorWeightedObjective,
    VfmRegionOfInterest,
    MetricSBVF,
    OptimiserLeastSquares,
    VectorFirstResultPassthrough,
    EquilibriumGapMetric
)

INPUTS_PATH = Path(
    "/media/data/3_Resources/gr91-weld-dic-results/wdbn1/pyvale-input/"
    "vfm-input-data_2026-08-12_15-43"
)

OUTPUT_PLOT_PATH = Path(__file__).resolve().parent / "call_vfm_sw_refine_results.png"


def main() -> None:

    experiment_data = ExperimentData.load_from_file(INPUTS_PATH / "experiment_data.yaml")

    # Define constitutive parameters associated with constitutive law
    constitutive_law = IsotropicVonMisesElastoplasticity(HardeningLinear())
    parameter_map_size = np.array(experiment_data.specimen_geometry.x.shape)
    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            210_000, 150_000, 250_000, parameter_map_size
        ),
        "poissons_ratio": ConstitutiveParameter(
            0.3, 0.2, 0.4, parameter_map_size
        ),
        "yield_strength": ConstitutiveParameter(
            200, 100, 2000, parameter_map_size
        ),
        "hardening_modulus": ConstitutiveParameter(
            3000, 1000, 15_000, parameter_map_size
        ),
    }


    # Define identification phases
    # Similar neighbouring slices are merged, and high-error slices are split,
    # once after the first solve.
    phases = [
        IdentificationPhase(
            spatial_parameterisations={
                "elastic_modulus": [SpatialParameterisationKnown()],
                "poissons_ratio": [SpatialParameterisationKnown()],
                "yield_strength": [SpatialParameterisationHomogeneous()],
                "hardening_modulus": [SpatialParameterisationKnown()],
            },
            metrics=[EquilibriumGapMetric()],       
            objective_function=VectorFirstResultPassthrough(),
            optimiser=OptimiserLeastSquares(),
        ),
    ]

    # Assemble the identification configuration
    identification = IdentificationConfig(
        constitutive_law=constitutive_law,
        parameters=parameters,
        phases=phases,
    )

    # Run identification, refine the shared support once, then solve again.
    vfm_result = run_identification(experiment_data, identification)
    print(vfm_result)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, param_name, title in zip(
        axes,
        ("yield_strength", "hardening_modulus"),
        ("Yield Strength", "Hardening Modulus"),
        strict=True,
    ):
        image = ax.imshow(vfm_result.parameter_maps[param_name], origin="lower", cmap="viridis")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
    fig.savefig(OUTPUT_PLOT_PATH, dpi=200)
    print(f"Saved slicewise refinement plot to {OUTPUT_PLOT_PATH}")
    if plt.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
