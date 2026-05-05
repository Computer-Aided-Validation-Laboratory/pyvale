import numpy as np

from pyvale.vfm.experiment_data import (
    BoundaryConditions,
    EEdge,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.identification import Identification
from pyvale.vfm.constitutive_laws.linear_hardening import LinearHardening
from pyvale.vfm.identification_phase import IdentificationPhase
from pyvale.vfm.metrics.sensitivity_based_virtual_fields import (
    SensitivityBasedVitualFieldsMetric,
)
from pyvale.vfm.parameter import ConstitutiveParameter
from pyvale.vfm.spatial_parameterisations.known import (
    KnownSpatialParameterisation
)
from pyvale.vfm.spatial_parameterisations.homogeneous import (
    HomogeneousSpatialParameterisation
)
from pyvale.vfm.vfm import vfm
from pyvale.vfm.optimisers.least_squares import LeastSquares


def main():
    specimen_geometry = SpecimenGeometry(
        np.load("inputs/x.npy"),
        np.load("inputs/y.npy"),
        np.load("inputs/specimen_mask.npy"),
        1.8,
        np.load("inputs/pixel_area.npy"),
    )

    boundary_conditions = BoundaryConditions(
        {
            EEdge.Top: EEdgeCondition.Traction,
            EEdge.Bottom: EEdgeCondition.Fixed,
            EEdge.Left: EEdgeCondition.Free,
            EEdge.Right: EEdgeCondition.Free,
        },
        np.load("inputs/force.npy"),
    )

    experiment_data = ExperimentData(
        np.load("inputs/strain.npy"),
        specimen_geometry,
        boundary_conditions,
        np.load("inputs/time.npy"),
    )

    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            190_000, 150_000, 250_000, np.array([113, 316])
        ),
        "poissons_ratio": ConstitutiveParameter(0.28, 0.2, 0.4, np.array([113, 316])),
        "yield_strength": ConstitutiveParameter(320, 100, 2000, np.array([113, 316])),
        "hardening_modulus": ConstitutiveParameter(
            3000, 1000, 10_000, np.array([113, 316])
        ),
    }

    phases = [
        IdentificationPhase(
            {
                "elastic_modulus": KnownSpatialParameterisation("elastic_modulus"),
                "poissons_ratio": KnownSpatialParameterisation("poissons_ratio"),
                "yield_strength": HomogeneousSpatialParameterisation("yield_strength"),
                "hardening_modulus": HomogeneousSpatialParameterisation("hardening_modulus"),
            },
            [(SensitivityBasedVitualFieldsMetric(np.array([3, 3])), 1.0)],
            LeastSquares(),
        )
    ]

    identification = Identification(LinearHardening(), parameters, phases)

    vfm_result = vfm(experiment_data, identification)


if __name__ == "__main__":
    main()
