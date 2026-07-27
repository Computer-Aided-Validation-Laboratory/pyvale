from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
from pyvale.vfm.identificationconfig import IdentificationConfig, IdentificationPhase
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.objectivefuncvector import VectorWeightedObjective
from pyvale.vfm.optimiserslicewiseindependent import SliceWiseIndependentLeastSquares
from pyvale.vfm.spatialparamknown import SpatialParameterisationKnown
from pyvale.vfm.spatialparamslicewise import (
    SliceConfig,
    SliceWiseSpatialParameterisation,
    build_slice_partition,
)
from pyvale.vfm.vfmregionofinterest import VfmRegionOfInterest


INPUTS_PATH = Path(__file__).resolve().parent / "rob-data" / "wdbn4-vfm-input-data-260629-1530"
SLICE_AXIS = "y"
NUM_SLICES = 20
PLOT_SLICE_PARTITION = True
PLOT_SLICE_INDEX = 0


def _print_coverage_diagnostic(slice_partition) -> None:
    mean_coverage = float(np.mean(slice_partition.coverage_fractions))
    min_slice_index = int(np.argmin(slice_partition.coverage_fractions))
    min_slice_coverage = float(slice_partition.coverage_fractions[min_slice_index])
    overall_coverage = float(np.sum(slice_partition.areas) / np.sum(slice_partition.geometric_areas))

    print("Slice coverage diagnostic:")
    print(f"  overall area coverage ratio: {overall_coverage:.6f}")
    print(f"  mean slice coverage ratio:   {mean_coverage:.6f}")
    print(f"  minimum slice coverage:      {min_slice_coverage:.6f} (slice {min_slice_index})")
    print("  per-slice coverage ratios:")
    for slice_index, coverage in enumerate(slice_partition.coverage_fractions):
        print(f"    slice {slice_index:>2d}: {float(coverage):.6f}")


def main() -> None:
    specimen_geometry = SpecimenGeometry(
        x=np.load(INPUTS_PATH / "x.npy"),
        y=np.load(INPUTS_PATH / "y.npy"),
        region_of_interest=VfmRegionOfInterest.from_yaml(INPUTS_PATH / "region_of_interest.yaml"),
        thickness=0.8,
        pixel_area=np.load(INPUTS_PATH / "pixel_area.npy"),
    )

    boundary_conditions = BoundaryConditions(
        EdgeConditions(
            min_x_edge=Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Free,
            ),
            max_x_edge=Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Free,
            ),
            min_y_edge=Edge(
                EEdgeCondition.Fixed,
                EEdgeCondition.Fixed,
            ),
            max_y_edge=Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Traction,
            ),
        ),
        np.load(INPUTS_PATH / "force.npy"),
    )

    experiment_data = ExperimentData(
        np.load(INPUTS_PATH / "strain.npy"),
        specimen_geometry,
        boundary_conditions,
        np.load(INPUTS_PATH / "time.npy"),
    )

    parameter_map_size = np.array(specimen_geometry.x.shape)
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

    # slice_config = 

    # slice_partition = build_slice_partition(
    #         specimen_geometry,
    #         slice_config=slice_config,
    #         plot_diagnostic=True,
    #         diagnostic_slice_index=PLOT_SLICE_INDEX,
    #     )
    # _print_coverage_diagnostic(slice_partition)

    phases = [
        IdentificationPhase(
            spatial_parameterisations={
                "elastic_modulus": [SpatialParameterisationKnown()],
                "poissons_ratio": [SpatialParameterisationKnown()],
                "yield_strength": [SliceWiseSpatialParameterisation(slice_config = SliceConfig(axis=SLICE_AXIS, num_slices=NUM_SLICES))],
                "hardening_modulus": [SliceWiseSpatialParameterisation(slice_config = SliceConfig(axis=SLICE_AXIS, num_slices=NUM_SLICES))],
            },
            metrics=[SliceWiseForceReconstructionMetric( slice_config=SliceConfig(axis=SLICE_AXIS, num_slices=NUM_SLICES))],
            objective_function=VectorWeightedObjective(),
            optimiser=SliceWiseIndependentLeastSquares(),
        )
    ]

    identification = IdentificationConfig(
        IsotropicVonMisesElastoplasticity(
            HardeningLinear()
        ),
        parameters,
        phases,
    )

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
    plt.show()


if __name__ == "__main__":
    main()
