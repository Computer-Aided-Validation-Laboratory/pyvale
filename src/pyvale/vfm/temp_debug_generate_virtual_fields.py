import pickle
from pyvale.vfm.virtual_fields_mesh import (
    generate_virtual_fields_mesh,
    plot_virtual_fields_mesh,
)


with open("tmp/debug_virtual_fields_mesh_case.pkl", "rb") as f:
    data = pickle.load(f)

x = data["x"]
y = data["y"]
specimen_mask = data["specimen_mask"]
boundary_conditions = data["boundary_conditions"]
mesh_size = data["mesh_size"]


generate_virtual_fields_mesh(
    x,
    y,
    specimen_mask,
    boundary_conditions,
    mesh_size,
)
