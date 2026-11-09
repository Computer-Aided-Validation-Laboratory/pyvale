# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Inspect and manipulate SimData
===============================

Read a simulation into :class:`pyvale.dataio.SimData`, inspect every field and
select arrays for analysis. This example follows the quickstart and runs
independently using the packaged case00_HEX8 thermo-mechanical cube. Its native
units are metres, pascals, seconds and degrees Celsius; strain is dimensionless.
No figures or interactive input are required.
"""

from dataclasses import fields
from pathlib import Path

import numpy as np
import numpy.typing as npt

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh

# %%
# Run a small model so this notebook needs no output from an earlier example.
# To inspect your own existing Exodus output instead, set output_file to its Path
# and start at the next cell.
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8)

print(f"Packaged MOOSE input: {input_path}")
result: mh.RunResult = mh.MooseRunner().run(
    input_path,
    output_dir=mh.get_next_run_path(output_path / "ex01b_simdata"),
)

output_file: Path = result.output_file

# %%
# load_all_sim_data() reads every available field into memory. The with block
# closes the NetCDF file; the returned arrays remain usable after the block.
with mh.ExodusLoader(output_file) as loader:
    sim_data: io.SimData = loader.load_all_sim_data()

# %%
# SimData is a dataclass inspect its declared fields and types, then use 
# attributes such as sim_data.coords to access each field.
for field in fields(sim_data):
    print(f"{field.name}: {field.type}")

# %%
# Let N be the node count and T the number of saved time samples.
# time has shape (T,), coords has shape (N, 3), with columns x, y, z.
# Even a 2D model stores three coordinate columns. num_spat_dims gives the number
# of active spatial dimensions; mesh_type distinguishes volume/surface meshes.
# SimData may be partially populated: an omitted/unavailable field can be None.
print(f"Spatial dimensions: {sim_data.num_spat_dims}")
print(f"Mesh type: {sim_data.mesh_type}")
print(f"time: shape={sim_data.time.shape}, seconds={sim_data.time}")
print(f"coords: shape={sim_data.coords.shape}, first node (m)={sim_data.coords[0]}")

# %%
# Dictionary keys identify blocks, boundaries and physical variables. Print all
# keys and actual array shapes, rather than assuming which fields a deck saves.
#
# connect: 'connect1', 'connect2', ... identify mesh blocks. Each array has shape
# (E_b, nodes_per_element), with one element per row. The default loader normalises
# connectivity to ZERO-BASED node indices suitable for indexing coords directly.
#
# side_sets: (boundary_name, 'node') -> node numbers, shape (N_boundary,);
# (boundary_name, 'elem') -> element numbers, shape (E_boundary,).
# IMPORTANT: ExodusLoader retains ONE-BASED Exodus numbers in side_sets even
# though connect is normalised. Convert node numbers before NumPy indexing.
#
# node_vars: 'disp_x', 'temperature', ... -> shape (N, T).
# Material outputs such as strain may also be stored here, depending on the deck.
#
# elem_vars: (variable_name, block_number), e.g. ('stress_yy', 1), -> (E_b, T).
# The block number in this key is the loader's one-based block number, not an
# array index. This cube saves strains as nodal fields and has elem_vars=None.
#
# glob_vars: postprocessor names such as 'react_y_top' -> shape (T,).
# A dictionary can be None when that category was not requested or is absent.
for name in ("connect", "side_sets", "node_vars", "elem_vars", "glob_vars"):
    values = getattr(sim_data, name)
    print(f"\n{name}:")
    if values is None:
        print("  None (not loaded or not present in this output)")
    else:
        for key, array in values.items():
            print(f"  {key!r}: shape={array.shape}, dtype={array.dtype}")

# %%
# Select the final time column, or a complete history at one zero-based node.
# Basic slices usually return views: use copy() before modifying source values.

final_disp_y: npt.NDArray[np.float64] = sim_data.node_vars["disp_y"][:, -1]
node_index: int = 0
node_temperature: npt.NDArray[np.float64] = sim_data.node_vars["temperature"][
    node_index, :
]
final_disp_y_mm: npt.NDArray[np.float64] = final_disp_y.copy()
final_disp_y_mm *= 1000.0  # A derived array in mm; the original remains in m.

print(f"Final displacement (mm): {final_disp_y_mm}")
print(f"Node {node_index} temperature history (degrees Celsius): {node_temperature}")

# %%
# Connectivity already indexes coords directly; side-set node numbers need -1.
first_element_coords: npt.NDArray[np.float64] = sim_data.coords[
    sim_data.connect["connect1"][0]
]

top_node_indices: npt.NDArray[np.int64] = (
    sim_data.side_sets[("top", "node")].astype(np.int64) - 1
)

top_disp_y: npt.NDArray[np.float64] = final_disp_y[top_node_indices]
reaction_y: npt.NDArray[np.float64] = sim_data.glob_vars["react_y_top"]

print(f"First element coordinates (m):\n{first_element_coords}")
print(f"Mean final top displacement (m): {top_disp_y.mean()}")
print(f"Top reaction history (N): {reaction_y}")

# %%
# For large files, inspect the available keys before loading selected fields.
# get_read_config() returns a SimLoadConfig populated with all available names.
# A fresh config defaults to geometry/time but no variable selections. Explicitly
# select variables and time_inds; the time axis is retained even for one sample.
with mh.ExodusLoader(output_file) as loader:
    available: io.SimLoadConfig = loader.get_read_config()

    print(f"Available field selections:\n{available}")
    selection: io.SimLoadConfig = io.SimLoadConfig(
        node_vars=("disp_y",),
        glob_vars=("react_y_top",),
        sidesets=("top",),
        time_inds=slice(-1, None),
    )

    selected_data: io.SimData = loader.load_sim_data(selection)

print(f"Selected times (s): {selected_data.time}")
print(f"Selected nodal keys: {tuple(selected_data.node_vars)}")
print(f"Selected disp_y shape: {selected_data.node_vars['disp_y'].shape}")

# %%
# Save derived values for downstream analysis without changing the loaded SimData.
np.savez(
    result.output_dir / "simdata_inspection.npz",
    time_s=sim_data.time,
    final_disp_y_m=final_disp_y,
    final_disp_y_mm=final_disp_y_mm,
    top_node_indices=top_node_indices,
    top_disp_y_m=top_disp_y,
    reaction_y_n=reaction_y,
    selected_disp_y_m=selected_data.node_vars["disp_y"],
)
