
import numpy as np

from pyvale.vfm import (
    ConstitutiveParameter,
    ExperimentData,
    HardeningLinear,
    IdentificationConfig,
    IdentificationPhase,
    IsotropicVonMisesElastoplasticity,
    MetricSBVF,
    OptimiserLeastSquares,
    run_identification,
    SpatialParameterisationHomogeneous,
    VectorFirstResultPassthrough,
)


def main():
    experiment_data = ExperimentData.load_from_file(
        "/Users/chris/work/pyvale/vfm-input-data_2026-08-10_13-02/experiment_data.yaml"
    )

    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            190_000, 150_000, 250_000, np.array([101, 101])
        ),
        "poissons_ratio": ConstitutiveParameter(
            0.28, 0.2, 0.4, np.array([101, 101])
        ),
        "yield_strength": ConstitutiveParameter(
            320, 100, 1000, np.array([101, 101])
        ),
        "hardening_modulus": ConstitutiveParameter(
            3000, 1000, 10_000, np.array([101, 101])
        ),
    }

    phases = [
        IdentificationPhase(
            {
                "elastic_modulus": [SpatialParameterisationHomogeneous()],
                "poissons_ratio": [SpatialParameterisationHomogeneous()],
                "yield_strength": [SpatialParameterisationHomogeneous()],
                "hardening_modulus": [SpatialParameterisationHomogeneous()],
            },
            [
                MetricSBVF(np.array([15, 15]))
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

    vfm_result.save_to_yaml()


if __name__ == "__main__":
    main()
