import gmsh
import os
from pathlib import Path

# ==========================================
# IMPORTANT USER INPUTS
# ==========================================

OUT_DIR = PARENT_DIR = Path(__file__).resolve().parent / "beam"/"exp_coarse"

# Beam dimensions
# Box(1) = {0, 0, 0, 12, 1, 144}; // 12 mm wide, 1 mm thick, 144 mm long beam
# For convergence
LX = 12.0 # mm
LY = 48.0 # mm; length WITHOUT the T-head holding it
LZ = 0.9 # mm
# For experimental test
LX = 12.11 # mm
LY = 50.0 # mm; length WITHOUT the T-head holding it
LZ = 0.9 # mm

# Characteristic lengths for unstructured meshing
# THIS ACTUALLY AFFECTS THE REFINEMENT

# Coarse
MESH_SIZE_MIN = 0.5
MESH_SIZE_MAX = 0.5

# Med-fine
#MESH_SIZE_MIN = 0.35
#MESH_SIZE_MAX = 0.35

# Fine
#MESH_SIZE_MIN = 0.2
#MESH_SIZE_MAX = 0.2

# IMPORTANT:
# Number of NODES along edges parallel to each axis
# Number of first-order elements along each such edge is:
# ex = NX - 1, ey = NY - 1, ez = NZ - 1
NX = 13    # along x = 12 elements
NY = 145   # along y = 144 elements
NZ = 2     # along z = 1 element

# If True, force structured/transfinite meshing on all faces
USE_TRANSFINITE = True

# Quad recombination algorithm
RECOMB_ALGO = 1

def reset_model(name):
    gmsh.clear()
    gmsh.model.add(name)


def build_beam():
    # IMPORTANT:
    # Equivalent to:
    # SetFactory("OpenCASCADE");
    # Box(1) = {0, 0, 0, 12, 1, 144};
    gmsh.model.occ.addBox(0.0, 0.0, 0.0, LX, LY, LZ)
    gmsh.model.occ.synchronize()

    curves = gmsh.model.getEntities(1)
    surfaces = gmsh.model.getEntities(2)
    return curves, surfaces


def set_common_options():
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Format", 2)  # VTK legacy 

    # IMPORTANT:
    # These control target element size mainly for unstructured meshing
    gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE_MIN)
    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE_MAX)    


def edge_direction(bbox, tol=1e-9):
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    if dx > tol and dy < tol and dz < tol:
        return "x"
    if dy > tol and dx < tol and dz < tol:
        return "y"
    if dz > tol and dx < tol and dy < tol:
        return "z"
    return None


def apply_transfinite_to_beam(curves, surfaces):
    # IMPORTANT:
    # Assign transfinite node counts based on whether an edge is parallel to x, y, or z
    for dim, tag in curves:
        bbox = gmsh.model.getBoundingBox(dim, tag)
        direction = edge_direction(bbox)

        if direction == "x":
            gmsh.model.mesh.setTransfiniteCurve(tag, NX)
        elif direction == "y":
            gmsh.model.mesh.setTransfiniteCurve(tag, NY)
        elif direction == "z":
            gmsh.model.mesh.setTransfiniteCurve(tag, NZ)

    # IMPORTANT:
    # A box has 6 four-sided faces, so each face can be made transfinite.
    for dim, tag in surfaces:
        gmsh.model.mesh.setTransfiniteSurface(tag)


def generate_tri_mesh(order, filename):
    reset_model("beam_tri")
    curves, surfaces = build_beam()
    set_common_options()

    if USE_TRANSFINITE:
        apply_transfinite_to_beam(curves, surfaces)

    # IMPORTANT:
    # generate(2) -> surface mesh only
    gmsh.model.mesh.generate(2)

    # IMPORTANT:
    # order=1 -> TRI3
    # order=2 -> TRI6
    gmsh.model.mesh.setOrder(order)

    gmsh.write(os.path.join(OUT_DIR, filename))


def generate_quad_mesh(order, second_order_incomplete, filename):
    reset_model("beam_quad")
    curves, surfaces = build_beam()
    set_common_options()

    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", RECOMB_ALGO)

    # IMPORTANT:
    # Recombine all 6 faces so they become quadrilateral surface meshes.
    for dim, tag in surfaces:
        gmsh.model.mesh.setRecombine(dim, tag)

    if USE_TRANSFINITE:
        apply_transfinite_to_beam(curves, surfaces)

    gmsh.model.mesh.generate(2)

    # IMPORTANT:
    # second_order_incomplete = 1 -> QUAD8
    # second_order_incomplete = 0 -> QUAD9
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", second_order_incomplete)

    # IMPORTANT:
    # order=1 -> QUAD4
    # order=2 -> QUAD8 or QUAD9 depending on Mesh.SecondOrderIncomplete
    gmsh.model.mesh.setOrder(order)

    gmsh.write(os.path.join(OUT_DIR, filename))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    generate_tri_mesh(order=1, filename="beam_surface_TRI3.vtk")
    generate_tri_mesh(order=2, filename="beam_surface_TRI6.vtk")

    generate_quad_mesh(order=1, second_order_incomplete=0, filename="beam_surface_QUAD4.vtk")
    generate_quad_mesh(order=2, second_order_incomplete=1, filename="beam_surface_QUAD8.vtk")
    generate_quad_mesh(order=2, second_order_incomplete=0, filename="beam_surface_QUAD9.vtk")

    gmsh.finalize()


if __name__ == "__main__":
    gmsh.initialize()
    main()