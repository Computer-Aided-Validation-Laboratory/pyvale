# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Quickstart virtual fields method
================================================================================

This is a quick example with minimal explanation to get users familiar with the
overall workflow for the `pyvale` virtual fields method (VFM) engine. The VFM
identifies the parameters of a material constitutive model directly from
full-field strain measurements (e.g. from digital image correlation) and the
global reaction force measured during a mechanical test.

The general workflow for the VFM engine in pyvale is:

1. Process raw solver / experiment output into the ``ExperimentData`` format;
2. Load the processed ``ExperimentData`` from file;
3. Set up the identification (constitutive model, initial parameters, phases);
4. Run the identification and inspect the identified parameters.

Users with experience in inverse identification will recognise the workflow as:
pre-processing the measured fields; defining the model and virtual fields; then
running an optimiser that drives the modelled internal virtual work towards the
measured external virtual work.
"""

from pathlib import Path

import numpy as np

# pyvale imports
import pyvale.vfm as vfm

# %%
# 1. Process raw data into the ExperimentData format
# --------------------------------------------------
# The input data processor loads raw field output from a solver (MOOSE or
# ANSYS) or an experiment, interpolates it onto a regular grid, validates it,
# writes diagnostic images, and saves a portable ``ExperimentData`` bundle
# (an ``experiment_data.yaml`` alongside ``.npy`` field arrays).
#
# Here we describe a MOOSE elasto-plastic solve with a ``MooseConfig``. Point
# ``exodus_file_path`` at your own solver output. The ``edge_conditions``
# describe the mechanical boundary condition on each of the four specimen edges:
# here the bottom edge is fixed and the top edge is pulled in the y-direction
# (a known traction), with the left and right edges free.
edge_conditions = vfm.EdgeConditions(
    min_x_edge=vfm.Edge(x=vfm.EEdgeCondition.Free, y=vfm.EEdgeCondition.Free),
    max_x_edge=vfm.Edge(x=vfm.EEdgeCondition.Free, y=vfm.EEdgeCondition.Free),
    min_y_edge=vfm.Edge(x=vfm.EEdgeCondition.Fixed, y=vfm.EEdgeCondition.Fixed),
    max_y_edge=vfm.Edge(x=vfm.EEdgeCondition.Free, y=vfm.EEdgeCondition.Traction),
)

input_config: vfm.MooseConfig = vfm.MooseConfig(
    exodus_file_path="path/to/your/moose_output.e",
    height=50.0,       # specimen height (mm)
    width=50.0,        # specimen width (mm)
    thickness=1.0,     # out-of-plane thickness (mm)
    grid_divs=101,     # interpolation grid divisions per axis
    edge_conditions=edge_conditions,
)

# ``process_input_data`` returns the path to the saved ``experiment_data.yaml``.
# It creates a timestamped run directory under ``output_root``.
experiment_data_file: Path = vfm.process_input_data(input_config, output_root=".")

# %%
# 2. Load the processed experiment data from file
# -----------------------------------------------
# Once the data has been processed it can be reloaded at any time from the saved
# ``experiment_data.yaml``, decoupling the (potentially slow) pre-processing
# from the identification itself.
experiment_data: vfm.ExperimentData = vfm.ExperimentData.load_from_file(
    experiment_data_file
)

# The parameter maps span the same grid as the measured strain field.
map_size: np.ndarray = np.array(
    experiment_data.specimen_geometry.x.shape, dtype=np.uint32
)

# %%
# 3. Set up the identification
# ----------------------------
# Choose the constitutive model to identify. Here we use isotropic von Mises
# elasto-plasticity with linear (bilinear) hardening, which has four parameters:
# ``elastic_modulus``, ``poissons_ratio``, ``yield_strength`` and
# ``hardening_modulus``.
constitutive_law = vfm.IsotropicVonMisesElastoplasticity(vfm.HardeningLinear())

# Provide an initial guess plus lower/upper bounds for each parameter. The
# optimiser searches within these bounds.
parameters = {
    "elastic_modulus": vfm.ConstitutiveParameter(
        200_000.0, 100_000.0, 300_000.0, map_size  # MPa
    ),
    "poissons_ratio": vfm.ConstitutiveParameter(
        0.3, 0.1, 0.5, map_size
    ),
    "yield_strength": vfm.ConstitutiveParameter(
        250.0, 100.0, 1000.0, map_size  # MPa
    ),
    "hardening_modulus": vfm.ConstitutiveParameter(
        1000.0, 500.0, 10_000.0, map_size  # MPa
    ),
}

# An identification phase pairs a spatial parameterisation for each parameter
# with the virtual-work metric, objective function and optimiser used to solve
# it. Here every parameter is assumed spatially homogeneous (a single value
# across the specimen), evaluated with the sensitivity-based virtual fields
# (SBVF) metric on a 15x15 virtual mesh.
phases = [
    vfm.IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [vfm.SpatialParameterisationHomogeneous()],
            "poissons_ratio": [vfm.SpatialParameterisationHomogeneous()],
            "yield_strength": [vfm.SpatialParameterisationHomogeneous()],
            "hardening_modulus": [vfm.SpatialParameterisationHomogeneous()],
        },
        metrics=[vfm.MetricSBVF(np.array([15, 15], dtype=np.uint32))],
        objective_function=vfm.VectorFirstResultPassthrough(),
        optimiser=vfm.OptimiserLeastSquares(),
    )
]

identification_config = vfm.IdentificationConfig(
    constitutive_law=constitutive_law,
    parameters=parameters,
    phases=phases,
)

# %%
# 4. Run the identification
# -------------------------
# ``run_identification`` returns the identified parameters as a mapping of
# parameter name to ``ConstitutiveParameter``. For a homogeneous
# parameterisation every entry of the parameter ``map`` holds the same
# identified value, so the mean recovers the scalar result.
identified_parameters = vfm.run_identification(
    experiment_data, identification_config
)

for name, parameter in identified_parameters.items():
    print(f"{name} = {np.nanmean(parameter.map):.4f}")
