# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Prepare Virtual Fields Method input data
========================================

Build a :class:`~pyvale.vfm.VfmInputData` object from a strain history,
specimen geometry, and a measured force history. This example uses a small
rectangular tensile test with illustrative measurements so you can see every
input you would supply for your own experiment.
"""

import numpy as np
import numpy.typing as npt

import pyvale.vfm as vfm
from pyvale.vfm.validation import run_validation


# %%
# 1. Supply strain and time histories
# -----------------------------------
# Provide one time value per DIC frame in strictly increasing order. The
# strain array has shape ``(n_times, 3, ny, nx)``, with components ordered
# ``[strain_xx, strain_yy, strain_xy]``. Strain is dimensionless. Use tensor
# shear for the third component, so divide engineering shear by two if that
# is what your DIC software exports.
#
# Here we broadcast a uniform strain to every grid point. For measured data,
# each frame should contain the strain measured at each individual point.
timesteps: npt.NDArray[np.float64] = np.array([0.0, 0.5, 1.0], dtype=np.float64)
strain_by_frame: npt.NDArray[np.float64] = np.array(
    [[0.0, 0.0, 0.0], [0.0005, -0.00015, 0.0], [0.001, -0.0003, 0.0]],
    dtype=np.float64,
)
ny, nx = 2, 4
strain: npt.NDArray[np.float64] = np.broadcast_to(
    strain_by_frame[:, :, np.newaxis, np.newaxis],
    (timesteps.size, 3, ny, nx),
).copy()

# %%
# 2. Describe the specimen geometry
# ---------------------------------
# Supply the reference coordinates ``x`` and ``y`` for the same points as the
# strain array. Both arrays have shape ``(ny, nx)`` and use millimetres.
# VFM expects a uniform axis-aligned grid with x increasing across columns
# and y increasing down rows. Each grid point has an associated area in mm².
#
# The ROI is defined in the same physical coordinates. Here the points are
# at the centres of cells that tile the specimen. For a specimen with a hole
# or cutout, exclude that region from the ROI. Strain must be finite inside
# the ROI in every frame. Unmeasured points outside it can be stored as NaN.
x_axis: npt.NDArray[np.float64] = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
y_axis: npt.NDArray[np.float64] = np.array([0.5, 1.5], dtype=np.float64)
x, y = np.meshgrid(x_axis, y_axis)
delta_x = x_axis[1] - x_axis[0]
delta_y = y_axis[1] - y_axis[0]
pixel_area: npt.NDArray[np.float64] = np.full(
    x.shape, delta_x * delta_y, dtype=np.float64
)

region_of_interest = vfm.VfmRegionOfInterest.from_definition(
    vfm.roi.RoiDefinition(
        shapes=(
            vfm.roi.RoiShape(
                shape_type="rectangle",
                index=0,
                is_cutting=False,
                rectangle=(0.0, 0.0, 4.0, 2.0),
            ),
        ),
    ),
)
specimen_geometry = vfm.SpecimenGeometry(
    x=x,
    y=y,
    pixel_area=pixel_area,
    thickness=1.0,  # mm
    region_of_interest=region_of_interest,
)

# %%
# 3. Supply the applied force and edge conditions
# ----------------------------------------------
# Provide a force array of shape ``(n_times, 2)`` in newtons, ordered
# ``[Fx, Fy]``. Each force row must describe the same instant as its DIC frame.
# Here the load acts along x and the transverse force is zero.
#
# Specify the condition in each direction for all four specimen edges. In
# this example the left edge is fixed along x, the right edge carries the
# x load, and the other edge directions are free.
force: npt.NDArray[np.float64] = np.array(
    [[0.0, 0.0], [200.0, 0.0], [400.0, 0.0]], dtype=np.float64
)
free_edge = vfm.Edge(x=vfm.EEdgeCondition.Free, y=vfm.EEdgeCondition.Free)
edge_conditions = vfm.EdgeConditions(
    min_x_edge=vfm.Edge(x=vfm.EEdgeCondition.Fixed, y=vfm.EEdgeCondition.Free),
    max_x_edge=vfm.Edge(x=vfm.EEdgeCondition.Traction, y=vfm.EEdgeCondition.Free),
    min_y_edge=free_edge,
    max_y_edge=free_edge,
)
boundary_conditions = vfm.BoundaryConditions(
    edge_conditions=edge_conditions,
    force=force,
)

# %%
# 4. Assemble and validate the input
# ----------------------------------
# The input object computes the time increments from the supplied times.
# Validate it before configuring identification. If your DIC and load-cell
# data are already in ``ExpData`` containers, ``vfm.build_vfm_input_data()``
# can assemble the same object from a regular grid and matching timestamps.
vfm_input = vfm.VfmInputData(
    strain=strain,
    specimen_geometry=specimen_geometry,
    boundary_conditions=boundary_conditions,
    timesteps=timesteps,
)
run_validation(vfm_input)

print("Strain (frames, components, rows, columns):", vfm_input.strain.shape)
print("Reference grid (rows, columns):", vfm_input.specimen_geometry.x.shape)
print("Force (frames, directions):", vfm_input.boundary_conditions.force.shape)
print("Frame times [s]:", vfm_input.timesteps)

# %%
# For identification, supply the constitutive law and decide which material
# parameters to fit. The homogeneous identification example shows the initial
# guesses and bounds. It takes the parameter-map shape from the specimen
# grid. The identification returns parameter maps on that grid. Any
# reference values in the later examples are used to assess recovery from
# synthetic measurements.
