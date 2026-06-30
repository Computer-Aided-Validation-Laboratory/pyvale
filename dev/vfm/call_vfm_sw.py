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
from pyvale.vfm.hardening import LinearHardening
from pyvale.vfm.identification import Identification, IdentificationPhase
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.objectivefuncvector import VectorFirstResultPassthrough
from pyvale.vfm.optimiserslicewiseindependent import SliceWiseIndependentLeastSquares
from pyvale.vfm.spatialparamslicewise import (
    SliceConfig,
    SliceWiseSpatialParameterisation,
    build_slice_partition,
)
from pyvale.vfm.spatialparamknown import KnownSpatialParameterisation
from pyvale.vfm.vfm import run_identification
from pyvale.vfm.vfmregionofinterest import VfmRegionOfInterest





inputs_path =Path(__file__).resolve().parent / "rob-data" / "wdbn4-vfm-input-data-260629-1530"

def main():

    
    specimen_geometry = SpecimenGeometry(
        x = np.load(inputs_path / "x.npy"),
        y = np.load(inputs_path / "y.npy"),
        region_of_interest = VfmRegionOfInterest.from_yaml(inputs_path / "region_of_interest.yaml"),
        thickness = 0.8,
        pixel_area = np.load(inputs_path / "pixel_area.npy"),
    )

    boundary_conditions = BoundaryConditions(
        EdgeConditions(
            min_x_edge=Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Free
            ),
            max_x_edge=Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Free
            ),
            min_y_edge=Edge(
                EEdgeCondition.Fixed,
                EEdgeCondition.Fixed
            ),
            max_y_edge=Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Traction
            )
        ),
        np.load(inputs_path / "force.npy"),
    )

    experiment_data = ExperimentData(
        np.load(inputs_path / "strain.npy"),
        specimen_geometry,
        boundary_conditions,
        np.load(inputs_path / "time.npy"),
    )

    # Define slice wise parameterisation
    slice_partition = build_slice_partition(
        specimen_geometry,
        slice_config=SliceConfig(axis="y", num_slices=20),
        plot_diagnostic=True,
        
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

    phases = [
        IdentificationPhase(
            {
                "elastic_modulus": KnownSpatialParameterisation(),
                "poissons_ratio": KnownSpatialParameterisation(),
                "yield_strength": SliceWiseSpatialParameterisation(slice_partition),
                "hardening_modulus": SliceWiseSpatialParameterisation(slice_partition),
            },
            [
                SliceWiseForceReconstructionMetric(slice_partition)
            ],
            VectorFirstResultPassthrough(),
            SliceWiseIndependentLeastSquares(),
        )
    ]

    identification = Identification(
        IsotropicVonMisesElastoplasticity(
            LinearHardening()
        ),
        parameters,
        phases
    )

    vfm_result = run_identification(experiment_data, identification)
    print(vfm_result)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, param_name, title in zip(
        axes,
        ("yield_strength", "hardening_modulus"),
        ("Yield Strength", "Hardening Modulus"),
        strict=True,
    ):
        image = ax.imshow(vfm_result[param_name].value, origin="lower", cmap="viridis")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
