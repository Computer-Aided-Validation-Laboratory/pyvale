import os
from pathlib import Path

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
from pyvale.vfm.optimiserleastsquares import LeastSquares
from pyvale.vfm.slicepartition import SliceConfig, build_slice_partition
from pyvale.vfm.spatialparamhomogeneous import HomogeneousSpatialParameterisation
from pyvale.vfm.spatialparamslicewise import SliceWiseSpatialParameterisation
from pyvale.vfm.spatialparamknown import KnownSpatialParameterisation
from pyvale.vfm.vfm import run_identification
from pyvale.vfm.vfmregionofinterest import VfmRegionOfInterest


def _resolve_inputs_path() -> Path:
    dataset_root = Path(__file__).resolve().parent / "rob-data" / "wdbn4-temporally-processed-data-260622-1404"
    prepared_candidates = sorted(dataset_root.glob("prepared-vfm-inputs-*"))
    return prepared_candidates[-1] if prepared_candidates else dataset_root


inputs_path = _resolve_inputs_path()

def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    specimen_geometry = SpecimenGeometry(
        np.load(inputs_path / "x.npy"),
        np.load(inputs_path / "y.npy"),
        VfmRegionOfInterest.from_yaml(inputs_path / "region_of_interest.yaml"),
        0.8,
        np.load(inputs_path / "pixel_area.npy"),
    )

    boundary_conditions = BoundaryConditions(
        EdgeConditions(
            Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Free
            ),
            Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Traction
            ),
            Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Free
            ),
            Edge(
                EEdgeCondition.Fixed,
                EEdgeCondition.Fixed
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
    parameter_map_size = np.array(specimen_geometry.x.shape)
    slice_partition = build_slice_partition(
        specimen_geometry,
        slice_config=SliceConfig(axis="y", num_slices=3),
    )

    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            190_000, 150_000, 250_000, parameter_map_size
        ),
        "poissons_ratio": ConstitutiveParameter(
            0.28, 0.2, 0.4, parameter_map_size
        ),
        "yield_strength": ConstitutiveParameter(
            320, 100, 1000, parameter_map_size
        ),
        "hardening_modulus": ConstitutiveParameter(
            3000, 1000, 10_000, parameter_map_size
        ),
    }

    phases = [
        IdentificationPhase(
            {
                "elastic_modulus": KnownSpatialParameterisation(),
                "poissons_ratio": KnownSpatialParameterisation(),
                "yield_strength": SliceWiseSpatialParameterisation(slice_partition),
                "hardening_modulus": HomogeneousSpatialParameterisation(),
            },
            [
                SliceWiseForceReconstructionMetric(slice_partition)
            ],
            VectorFirstResultPassthrough(),
            LeastSquares(),
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


if __name__ == "__main__":
    main()
