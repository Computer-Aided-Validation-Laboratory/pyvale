# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Example 1: homogeneous SBVF identification
===========================================

Identify four homogeneous elastoplastic parameters from a small synthetic
rectangular tensile test. The experiment is generated in memory, so this
example requires no external input files.
"""

import numpy as np

import pyvale.vfm as vfm
from pyvale.examples.vfm.synthetic_rectangular_tensile import (
    SyntheticTensileMaterial,
    build_synthetic_tensile_case,
)


# %%
# Build a homogeneous tensile experiment
# ---------------------------------------
# The helper constructs cell-centred geometry, a monotonic plane-stress strain
# history, and boundary forces consistent with the known stress field.
synthetic_case = build_synthetic_tensile_case(
    num_grid_columns=12,
    length=12.0,
    material=SyntheticTensileMaterial(yield_strengths=(250.0,)),
)
experiment_data = synthetic_case.experiment_data
map_size = np.asarray(
    experiment_data.specimen_geometry.x.shape,
    dtype=np.uint32,
)

# %%
# Configure homogeneous SBVF identification
# ------------------------------------------
# Each constitutive parameter has one homogeneous degree of freedom. SBVF
# supplies one virtual field per active degree of freedom and the least-squares
# optimiser drives the internal/external virtual-work residuals towards zero.
parameters = {
    "elastic_modulus": vfm.ConstitutiveParameter(
        180_000.0,
        150_000.0,
        260_000.0,
        map_size,
    ),
    "poissons_ratio": vfm.ConstitutiveParameter(
        0.25,
        0.15,
        0.45,
        map_size,
    ),
    "yield_strength": vfm.ConstitutiveParameter(
        220.0,
        150.0,
        350.0,
        map_size,
    ),
    "hardening_modulus": vfm.ConstitutiveParameter(
        5_000.0,
        1_000.0,
        15_000.0,
        map_size,
    ),
}

phase = vfm.IdentificationPhase(
    spatial_parameterisations={
        name: [vfm.SpatialParameterisationHomogeneous()]
        for name in parameters
    },
    metrics=[vfm.MetricSBVF(np.array([2, 3], dtype=np.uint32))],
    objective_function=vfm.VectorFirstResultPassthrough(),
    optimiser=vfm.OptimiserLeastSquares(max_evaluations=100),
    optimisation_newton_tolerance=1.0e-8,
)

identification_config = vfm.IdentificationConfig(
    constitutive_law=vfm.IsotropicVonMisesElastoplasticity(
        vfm.HardeningLinear(),
        error_tolerance=1.0e-10,
    ),
    parameters=parameters,
    phases=[phase],
)

# %%
# Run and inspect the result
# --------------------------
result = vfm.run_identification(experiment_data, identification_config)

for name, parameter_map in result.parameter_maps.items():
    identified = float(np.mean(parameter_map))
    reference = float(np.mean(synthetic_case.known_parameter_maps[name]))
    print(f"{name}: identified={identified:.6g}, reference={reference:.6g}")

solve_result = result.history.phases[0].solve_results[-1]
print(
    "solve: "
    f"success={solve_result.success}, "
    f"evaluations={solve_result.num_evaluations}, "
    f"runtime={solve_result.runtime_seconds:.3f} s"
)
