from pathlib import Path

import numpy as np
import pytest
from plots import plot_identification_diff
from rms import rms, root_mean_square_percentage_error

from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.hardening import HardeningLinear
from pyvale.vfm.identification import run_identification
from pyvale.vfm.identificationconfig import (
    IdentificationConfig,
    IdentificationPhase,
)
from pyvale.vfm.metricsbvf import MetricSBVF
from pyvale.vfm.objectivefuncvector import VectorFirstResultPassthrough
from pyvale.vfm.optimiserleastsquares import OptimiserLeastSquares
from pyvale.vfm.spatialparamhomogeneous import (
    SpatialParameterisationHomogeneous,
)

EXPERIMENT_DATA_FILE = (
    Path(__file__).parent
    / "input"
    / "hole2d_plas"
    / "experiment_data.yaml"
)

KNOWN_PARAMETERS_FILE = (
    Path(__file__).parent
    / "gold"
    / "hole2d_plas.npz"
)

PLOT_IDENTIFICATION_DIFF = False


@pytest.mark.skip(reason="tolerances need to be revised")
def test_end_to_end_homogeneous() -> None:
    experiment_data = ExperimentData.load_from_file(EXPERIMENT_DATA_FILE)

    # TODO: force is 1000x too large
    experiment_data.boundary_conditions.force *= 1e-3

    constitutive_law = IsotropicVonMisesElastoplasticity(HardeningLinear())

    parameter_map_size = np.array(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32
    )

    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            450_000, 100_000, 500_000, parameter_map_size
        ),
        "poissons_ratio": ConstitutiveParameter(
            0.45, 0.1, 0.5, parameter_map_size
        ),
        "yield_strength": ConstitutiveParameter(
            800, 100, 1000, parameter_map_size
        ),
        "hardening_modulus": ConstitutiveParameter(
            7000, 500, 10_000, parameter_map_size
        ),
    }

    metric = MetricSBVF(np.array([15, 15]))

    phases = [
        IdentificationPhase(
            {
                "elastic_modulus": [SpatialParameterisationHomogeneous()],
                "poissons_ratio": [SpatialParameterisationHomogeneous()],
                "yield_strength": [SpatialParameterisationHomogeneous()],
                "hardening_modulus": [SpatialParameterisationHomogeneous()],
            },
            [metric],
            VectorFirstResultPassthrough(),
            OptimiserLeastSquares(),
        )
    ]

    ident_config = IdentificationConfig(
        constitutive_law,
        parameters,
        phases
    )

    print("Running identification...")
    result = run_identification(experiment_data, ident_config)

    identified_maps = result.parameter_maps

    for name, param_map in identified_maps.items():
        print(f"{name} = {np.nanmean(param_map):.6f}")

    known_parameter_maps = dict(np.load(KNOWN_PARAMETERS_FILE))

    # ------------------------------------------------------------------
    # Test the result of the identification: compare the identified parameter
    # maps against the known parameter maps.
    # ------------------------------------------------------------------
    if PLOT_IDENTIFICATION_DIFF:
        plot_identification_diff(
            experiment_data.specimen_geometry.x,
            experiment_data.specimen_geometry.y,
            identified_maps,
            known_parameter_maps
        )

    # Per-parameter tolerances on the RMS of the absolute difference. The
    # hardening modulus is only weakly sensitive to the virtual fields and so
    # is identified less accurately than the other parameters.
    abs_diff_rms_tolerances = {
        "elastic_modulus": 400.0,
        "poissons_ratio": 1e-3,
        "yield_strength": 1.0,
        "hardening_modulus": 250.0,
    }

    for name in known_parameter_maps:
        abs_diff = np.abs(identified_maps[name] - known_parameter_maps[name])
        abs_diff_rms = rms(abs_diff)
        rmspe = root_mean_square_percentage_error(
            identified_maps[name], known_parameter_maps[name]
        )
        print(
            f"{name}: abs diff rms = {abs_diff_rms:.6f}, rmspe = {rmspe:.6f} %"
        )

        # The identified parameters should be close to the known parameters.
        assert abs_diff_rms < abs_diff_rms_tolerances[name]
        assert rmspe < 20.0
