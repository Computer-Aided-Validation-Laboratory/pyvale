import numpy as np
from load_sim_data import load_force, load_strain, load_timesteps
from plots import (
    _plot_identification_diff,
)
from utils import rms, root_mean_square_percentage_error

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

EXODUS_FILE_NAME = "out_hole2d_plas_32f.e"
GRID_DIVS = 101

PLATE_THICKNESS = 1e-3 # m

KNOWN_PARAMETERS = {
    "elastic_modulus": 200_000.0,  # MPa
    "poissons_ratio": 0.3,
    "yield_strength": 200.0,       # MPa
    "hardening_modulus": 1_000.0,  # MPa
}

PLOT_IDENTIFICATION_DIFF = False


def test_end_to_end_homogeneous() -> None:
    experiment_data = _setup_experiment_data()

    constitutive_law = IsotropicVonMisesElastoplasticity(LinearHardening())

    parameter_map_size = np.array([GRID_DIVS, GRID_DIVS], dtype=np.uint32)

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

    metric = SensitivityBasedVirtualFieldsMetric(np.array([15, 15]))

    phases = [
        IdentificationPhase(
            {
                "elastic_modulus": [HomogeneousSpatialParameterisation()],
                "poissons_ratio": [HomogeneousSpatialParameterisation()],
                "yield_strength": [HomogeneousSpatialParameterisation()],
                "hardening_modulus": [HomogeneousSpatialParameterisation()],
            },
            [metric],
            VectorFirstResultPassthrough(),
            LeastSquares(),
        )
    ]

    ident_config = IdentificationConfig(
        constitutive_law,
        parameters,
        phases
    )

    print("Running identification...")
    identified_parameters = run_identification(experiment_data, ident_config)

    for name, param in identified_parameters.items():
        print(f"{name} = {np.nanmean(param.map):.6f}")

    identified_maps = {
        name: param.map for name, param in identified_parameters.items()
    }

    known_parameter_maps = {
        name: np.full((GRID_DIVS, GRID_DIVS), value)
        for name, value in KNOWN_PARAMETERS.items()
    }

    # ------------------------------------------------------------------
    # Test the result of the identification: compare the identified parameter
    # maps against the known parameter maps.
    # ------------------------------------------------------------------
    if PLOT_IDENTIFICATION_DIFF:
        _plot_identification_diff(
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

    for name in KNOWN_PARAMETERS:
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


def _setup_experiment_data() -> ExperimentData:
    (x_grid, y_grid, strain) = load_strain(EXODUS_FILE_NAME, GRID_DIVS)
    force = load_force(EXODUS_FILE_NAME)
    timesteps = load_timesteps(EXODUS_FILE_NAME)

    specimen_mask = ~np.isnan(strain[0, 0, :, :])

    grid_element_area = (
        (x_grid[0, 1] - x_grid[0, 0]) * (y_grid[1, 0] - y_grid[0, 0])
    )

    specimen_geometry = SpecimenGeometry(
        x_grid,
        y_grid,
        specimen_mask,
        PLATE_THICKNESS,
        np.full_like(x_grid, grid_element_area, dtype=np.float64),
    )

    # seems to be an issue with FE input force data being 1000x too large
    force *= 1e-3

    boundary_conditions = BoundaryConditions(
        EdgeConditions(
            min_x_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Free),
            max_x_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Free),
            min_y_edge=Edge(x=EEdgeCondition.Fixed, y=EEdgeCondition.Fixed),
            max_y_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Traction),
        ),
        force
    )

    return ExperimentData(
        strain,
        specimen_geometry,
        boundary_conditions,
        timesteps,
    )
