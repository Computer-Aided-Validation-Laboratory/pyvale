import numpy as np
from pathlib import Path

from pyvale.vfm.experiment_data import (
    BoundaryConditions,
    EEdgeCondition,
    Edge,
    EdgeConditions,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.identification import Identification
from pyvale.vfm.constitutive_laws.linear_hardening import LinearHardening
from pyvale.vfm.identification import IdentificationPhase
from pyvale.vfm.metrics.virtual_fields.sensitivity_based_virtual_fields import (
    SensitivityBasedVirtualFieldsMetric,
)
from pyvale.vfm.constitutive_laws.constitutive_parameter import (
    ConstitutiveParameter,
)
from pyvale.vfm.objective_functions.residuals import Residuals
from pyvale.vfm.spatial_parameterisations.known import (
    KnownSpatialParameterisation,
)
from pyvale.vfm.spatial_parameterisations.homogeneous import (
    HomogeneousSpatialParameterisation,
)
from pyvale.vfm.vfm import vfm
from pyvale.vfm.optimisers.least_squares import LeastSquares

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
                SensitivityBasedVirtualFieldsMetric(
                    experiment_data.specimen_geometry.x,
                    experiment_data.specimen_geometry.y,
                    experiment_data.specimen_geometry.region_of_interest,
                    experiment_data.boundary_conditions.edge_conditions,
                    np.array([5, 10]),
                )
            ],
            Residuals(),
            LeastSquares(),
        )
    ]

    identification = Identification(LinearHardening(), parameters, phases)

    vfm_result = vfm(experiment_data, identification)
    print(vfm_result)


if __name__ == "__main__":
    main()
