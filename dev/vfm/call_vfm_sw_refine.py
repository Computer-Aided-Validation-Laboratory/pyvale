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
    SpecimenGeometry,
    SupportSlice,
    VectorWeightedObjective,
    VfmRegionOfInterest,
)


INPUTS_PATH = Path(__file__).resolve().parent / "rob-data" / "wdbn4-vfm-input-data-260629-1530"
SLICE_AXIS = "y"
NUM_SLICES = 40
MERGE_PARAMETER_TOLERANCE = 0.05
SPLIT_FORCE_ERROR_THRESHOLD = 0.1
OUTPUT_PLOT_PATH = Path(__file__).resolve().parent / "call_vfm_sw_refine_results.png"


def main() -> None:

    experiment_data = ExperimentData.load_from_file(INPUTS_PATH / "experiment_data.yaml")


    # # Prepare experiment data
    # specimen_geometry = SpecimenGeometry(
    #     x=np.load(INPUTS_PATH / "x.npy"),
    #     y=np.load(INPUTS_PATH / "y.npy"),
    #     region_of_interest=VfmRegionOfInterest.from_yaml(INPUTS_PATH / "region_of_interest.yaml"),
    #     thickness=0.8,
    #     pixel_area=np.load(INPUTS_PATH / "pixel_area.npy"),
    # )

    # boundary_conditions = BoundaryConditions(
    #     EdgeConditions(
    #         min_x_edge=Edge(
    #             EEdgeCondition.Free,
    #             EEdgeCondition.Free,
    #         ),
    #         max_x_edge=Edge(
    #             EEdgeCondition.Free,
    #             EEdgeCondition.Free,
    #         ),
    #         min_y_edge=Edge(
    #             EEdgeCondition.Fixed,
    #             EEdgeCondition.Fixed,
    #         ),
    #         max_y_edge=Edge(
    #             EEdgeCondition.Free,
    #             EEdgeCondition.Traction,
    #         ),
    #     ),
    #     np.load(INPUTS_PATH / "force.npy"),
    # )

    # experiment_data = ExperimentData(
    #     np.load(INPUTS_PATH / "strain.npy"),
    #     specimen_geometry,
    #     boundary_conditions,
    #     np.load(INPUTS_PATH / "time.npy"),
    # )

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

    # Shared object means shared evolution: both unknown parameters and the
    # slice-force metric use the same slice topology.
    shared_slice_support = SupportSlice(
        slice_config=SliceConfig(
            axis=SLICE_AXIS,
            num_slices=NUM_SLICES,
        ),
    )

    # Define identification phases
    # Similar neighbouring slices are merged, and high-error slices are split,
    # once after the first solve.
    phases = [
        IdentificationPhase(
            spatial_parameterisations={
                "elastic_modulus": [SpatialParameterisationKnown()],
                "poissons_ratio": [SpatialParameterisationKnown()],
                "yield_strength": [
                    SliceWiseSpatialParameterisation(
                        support=shared_slice_support
                    )
                ],
                "hardening_modulus": [
                    SliceWiseSpatialParameterisation(
                        support=shared_slice_support
                    )
                ],
            },
            metrics=[
                SliceWiseForceReconstructionMetric(
                    support=shared_slice_support
                )
            ],
            objective_function=VectorWeightedObjective(),
            optimiser=SliceWiseIndependentLeastSquares(),
            refinement_policy=SliceMergeSplitRefinement(
                target=shared_slice_support,
                merge_parameter_tolerance=MERGE_PARAMETER_TOLERANCE,
                split_error_threshold=SPLIT_FORCE_ERROR_THRESHOLD,
                max_refinements=1,
            ),
        )
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
        image = ax.imshow(vfm_result[param_name].map, origin="lower", cmap="viridis")
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
