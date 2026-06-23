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
from pyvale.vfm.identification import run_identification
from pyvale.vfm.identificationconfig import (
    IdentificationConfig,
    IdentificationPhase,
)
from pyvale.vfm.metricsbvf import SensitivityBasedVirtualFieldsMetric
from pyvale.vfm.objectivefuncvector import VectorFirstResultPassthrough
from pyvale.vfm.optimiserleastsquares import LeastSquares
from pyvale.vfm.spatialparamhomogeneous import (
    HomogeneousSpatialParameterisation,
)
from pyvale.vfm.spatialparamknown import KnownSpatialParameterisation

inputs_path = Path(__file__).resolve().parent / "inputs"

def main():
    specimen_geometry = SpecimenGeometry(
        np.load(inputs_path / "x.npy"),
        np.load(inputs_path / "y.npy"),
        np.load(inputs_path / "specimen_mask.npy"),
        1.8,
        np.load(inputs_path / "pixel_area.npy"),
    )

    boundary_conditions = BoundaryConditions(
        EdgeConditions(
            Edge(
                EEdgeCondition.Fixed,
                EEdgeCondition.Fixed
            ),
            Edge(
                EEdgeCondition.Traction,
                EEdgeCondition.Fixed
            ),
            Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Free
            ),
            Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Free
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

    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            190_000, 150_000, 250_000, np.array([113, 316])
        ),
        "poissons_ratio": ConstitutiveParameter(
            0.28, 0.2, 0.4, np.array([113, 316])
        ),
        "yield_strength": ConstitutiveParameter(
            320, 100, 1000, np.array([113, 316])
        ),
        "hardening_modulus": ConstitutiveParameter(
            3000, 1000, 10_000, np.array([113, 316])
        ),
    }

    phases = [
        IdentificationPhase(
            {
                "elastic_modulus": KnownSpatialParameterisation(),
                "poissons_ratio": KnownSpatialParameterisation(),
                "yield_strength": HomogeneousSpatialParameterisation(),
                "hardening_modulus": HomogeneousSpatialParameterisation(),
            },
            [
                SensitivityBasedVirtualFieldsMetric(np.array([5, 10]))
            ],
            VectorFirstResultPassthrough(),
            LeastSquares(),
        )
    ]

    identification_config = IdentificationConfig(
        IsotropicVonMisesElastoplasticity(
            LinearHardening()
        ),
        parameters,
        phases
    )

    vfm_result = run_identification(experiment_data, identification_config)
    print(vfm_result)


if __name__ == "__main__":
    main()
