from pathlib import Path

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
    MetricSBVF,
    OptimiserLeastSquares,
    run_identification,
    SpatialParameterisationBasisFunction,
    SpatialParameterisationHomogeneous,
    SpatialParameterisationKnown,
    SpecimenGeometry,
    VectorFirstResultPassthrough,
    VfmRegionOfInterest,
)

inputs_path = Path(__file__).resolve().parent / "inputs"

def main():
    specimen_geometry = SpecimenGeometry(
        np.load(inputs_path / "x.npy"),
        np.load(inputs_path / "y.npy"),
        VfmRegionOfInterest.from_yaml(inputs_path / "region_of_interest.yaml"),
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


    h, w = 113, 316

    i, j = np.indices((h, w))

    sigma = 20.0
    amplitude = 100

    # normalize coordinates to physical grid
    # line from (h-1, 0) → (0, w-1)

    # direction vector of diagonal
    di = -(h - 1)
    dj = (w - 1)

    # point on line (bottom-left)
    i0, j0 = h - 1, 0

    # perpendicular distance from each grid point to line
    dist = np.abs(dj*(i - i0) - di*(j - j0)) / np.sqrt(di**2 + dj**2)

    y = amplitude * np.exp(-(dist**2) / (2 * sigma**2))

    parameters = {
        # "elastic_modulus": ConstitutiveParameter(
        #     190_000, 150_000, 250_000, np.array([113, 316])
        # ),
        "elastic_modulus": ConstitutiveParameter(
            y, 150_000, 250_000
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
                "elastic_modulus": [SpatialParameterisationKnown()],
                "elastic_modulus": [
                    SpatialParameterisationBasisFunction(
                        experiment_data.specimen_geometry.x,
                        experiment_data.specimen_geometry.y,
                    )
                ],
                "poissons_ratio": [SpatialParameterisationKnown()],
                "yield_strength": [SpatialParameterisationHomogeneous()],
                "hardening_modulus": [SpatialParameterisationHomogeneous()],
            },
            [
                MetricSBVF(np.array([5, 10]))
            ],
            VectorFirstResultPassthrough(),
            OptimiserLeastSquares(),
        )
    ]

    identification_config = IdentificationConfig(
        IsotropicVonMisesElastoplasticity(
            HardeningLinear()
        ),
        parameters,
        phases
    )

    vfm_result = run_identification(experiment_data, identification_config)
    print(vfm_result)


if __name__ == "__main__":
    main()
