import gmsh
import os
from pathlib import Path

# ============================================================
# IMPORTANT USER INPUTS
# ============================================================

OUT_DIR = PARENT_DIR = Path(__file__).resolve().parent / "rectangular-box"
# ----------------------------
# Outer box dimensions
# ----------------------------
OUTER_SIDE = 48.0 # mm, outer box side length in x and z
OUTER_HEIGHT = 95.0 # mm, outer box height in y

# Base location of outer box
Y_BASE_OUTER = 0.0

# ----------------------------
# Inner void dimensions (to make hollowed box with thick walls)
# ----------------------------
TRIM_VALUE = 4.0 # mm, inner height = OUTER_HEIGHT - TRIM_VALUE
GAP_WIDTH = 5.0 # mm, inner side = OUTER_SIDE - GAP_WIDTH

INNER_SIDE = OUTER_SIDE - GAP_WIDTH
INNER_HEIGHT = OUTER_HEIGHT - TRIM_VALUE

# ----------------------------
# Fill dimensions (water, etc.)
# ----------------------------
# Fill needs to overlap slightly with the inner void to make nested dielectrics work
FILL_OVERLAP_SCALAR = 1.05   # IMPORTANT: > 1 makes fill larger than inner box
FILL_BOTTOM_BLEND = 0.8 # Determines where between y0_inner and y0_outer our fill mesh is located for overlap
WATER_LEVEL_OFFSET = 5.0     # height_fill = INNER_HEIGHT - WATER_LEVEL_OFFSET

FILL_SIDE = INNER_SIDE * FILL_OVERLAP_SCALAR
if FILL_SIDE > OUTER_SIDE:
    raise ValueError("Fill side can't be greater than the outer box side length.")

FILL_HEIGHT = INNER_HEIGHT - WATER_LEVEL_OFFSET

# ----------------------------
# Mesh density controls
# ----------------------------
# IMPORTANT:
# They are node counts on edges, so element counts are one less
NX = 25
NY = 49
NZ = 25

# Global characteristic lengths - TOGGLE ELEMENT COUNT
# Coarse
#MESH_SIZE_MIN = 1.0
#MESH_SIZE_MAX = 3.0

# Med-fine
MESH_SIZE_MIN = 0.5
MESH_SIZE_MAX = 1.5

# Fine
#MESH_SIZE_MIN = 0.25
#MESH_SIZE_MAX = 0.75

# IMPORTANT:
# Safer default after boolean cuts: avoid forcing transfinite surfaces
USE_TRANSFINITE_CURVES_ONLY = True

RECOMB_ALGO = 1

# ============================================================
# DERIVED GEOMETRY
# ============================================================

# Outer box centered in x-z, with base at Y_BASE_OUTER
x0_outer = -OUTER_SIDE / 2.0
y0_outer = Y_BASE_OUTER
z0_outer = -OUTER_SIDE / 2.0

# Inner box centered in x-z and aligned at the TOP in y
y_top_outer = y0_outer + OUTER_HEIGHT
x0_inner = -INNER_SIDE / 2.0
y0_inner = y_top_outer - INNER_HEIGHT
z0_inner = -INNER_SIDE / 2.0

# Fill box aligned similarly
x0_fill = -FILL_SIDE / 2.0
y0_fill = y0_outer + FILL_BOTTOM_BLEND * (y0_inner - y0_outer)
z0_fill = -FILL_SIDE / 2.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reset_model(name):
    gmsh.clear()
    gmsh.model.add(name)


def set_common_options():
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.AbortOnError", 2)
    gmsh.option.setNumber("Mesh.SaveAll", 1)

    # IMPORTANT:
    # Export directly as legacy VTK
    gmsh.option.setNumber("Mesh.Format", 2)

    gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE_MIN)
    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE_MAX)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", RECOMB_ALGO)


def classify_edge_direction(dim, tag, tol=1e-9):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
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


def apply_transfinite_curve_constraints():
    # IMPORTANT:
    # Only set transfinite constraints on curves
    # This is safer than forcing transfinite surfaces after OCC boolean cuts
    for dim, tag in gmsh.model.getEntities(1):
        direction = classify_edge_direction(dim, tag)
        if direction == "x":
            gmsh.model.mesh.setTransfiniteCurve(tag, NX)
        elif direction == "y":
            gmsh.model.mesh.setTransfiniteCurve(tag, NY)
        elif direction == "z":
            gmsh.model.mesh.setTransfiniteCurve(tag, NZ)


def set_surface_physical_group_from_volumes(volume_dimtags, name):
    boundary = gmsh.model.getBoundary(volume_dimtags, oriented=False, recursive=False)
    surf_tags = [tag for dim, tag in boundary if dim == 2]

    if surf_tags:
        pg = gmsh.model.addPhysicalGroup(2, surf_tags)
        gmsh.model.setPhysicalName(2, pg, name)

    return surf_tags

# ============================================================
# GEOMETRY BUILDERS
# ============================================================

def build_hollow_box_geometry():
    # Outer box: centered in x-z, height in y
    outer = gmsh.model.occ.addBox(
        x0_outer, y0_outer, z0_outer,
        OUTER_SIDE, OUTER_HEIGHT, OUTER_SIDE
    )

    # Inner box: centered in x-z, top-aligned in y
    inner = gmsh.model.occ.addBox(
        x0_inner, y0_inner, z0_inner,
        INNER_SIDE, INNER_HEIGHT, INNER_SIDE
    )

    # IMPORTANT:
    # Hollow shell = outer - inner
    out_dimtags, _ = gmsh.model.occ.cut(
        [(3, outer)],
        [(3, inner)],
        removeObject=True,
        removeTool=True
    )

    gmsh.model.occ.synchronize()
    return out_dimtags


def build_fill_box_geometry():
    # Independent fill box
    fill = gmsh.model.occ.addBox(
        x0_fill, y0_fill, z0_fill,
        FILL_SIDE, FILL_HEIGHT, FILL_SIDE
    )

    gmsh.model.occ.synchronize()
    return [(3, fill)]

# ============================================================
# MESH EXPORT HELPERS
# ============================================================

def make_tri3_surface_mesh(geometry_builder, filename, physical_name):
    reset_model(physical_name + "_tri3")
    set_common_options()

    vols = geometry_builder()
    surf_tags = set_surface_physical_group_from_volumes(vols, physical_name)

    if USE_TRANSFINITE_CURVES_ONLY:
        apply_transfinite_curve_constraints()

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)   # TRI3

    gmsh.write(os.path.join(OUT_DIR, filename))


def make_tri6_surface_mesh(geometry_builder, filename, physical_name):
    reset_model(physical_name + "_tri6")
    set_common_options()

    vols = geometry_builder()
    surf_tags = set_surface_physical_group_from_volumes(vols, physical_name)

    if USE_TRANSFINITE_CURVES_ONLY:
        apply_transfinite_curve_constraints()

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)   # TRI6

    gmsh.write(os.path.join(OUT_DIR, filename))


def make_quad_surface_mesh(order, second_order_incomplete, geometry_builder, filename, physical_name):
    reset_model(physical_name + "_quad")
    set_common_options()

    vols = geometry_builder()
    surf_tags = set_surface_physical_group_from_volumes(vols, physical_name)

    # IMPORTANT:
    # Recombine surface triangles into quads
    for s in surf_tags:
        gmsh.model.mesh.setRecombine(2, s)

    if USE_TRANSFINITE_CURVES_ONLY:
        apply_transfinite_curve_constraints()

    gmsh.model.mesh.generate(2)

    # IMPORTANT:
    # 1 -> QUAD8, 0 -> QUAD9 when order = 2
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", second_order_incomplete)
    gmsh.model.mesh.setOrder(order)

    gmsh.write(os.path.join(OUT_DIR, filename))


def export_all_for_geometry(geometry_builder, prefix, physical_name):
    make_tri3_surface_mesh(
        geometry_builder,
        f"{prefix}_TRI3.vtk",
        physical_name
    )

    make_tri6_surface_mesh(
        geometry_builder,
        f"{prefix}_TRI6.vtk",
        physical_name
    )

    make_quad_surface_mesh(
        order=1,
        second_order_incomplete=0,
        geometry_builder=geometry_builder,
        filename=f"{prefix}_QUAD4.vtk",
        physical_name=physical_name
    )

    make_quad_surface_mesh(
        order=2,
        second_order_incomplete=1,
        geometry_builder=geometry_builder,
        filename=f"{prefix}_QUAD8.vtk",
        physical_name=physical_name
    )

    make_quad_surface_mesh(
        order=2,
        second_order_incomplete=0,
        geometry_builder=geometry_builder,
        filename=f"{prefix}_QUAD9.vtk",
        physical_name=physical_name
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    export_all_for_geometry(
        geometry_builder=build_hollow_box_geometry,
        prefix="tank_surface",
        physical_name="hollow_box_surface"
    )

    export_all_for_geometry(
        geometry_builder=build_fill_box_geometry,
        prefix="fill_surface",
        physical_name="box_fill_surface"
    )

    gmsh.finalize()


if __name__ == "__main__":
    gmsh.initialize()
    main()