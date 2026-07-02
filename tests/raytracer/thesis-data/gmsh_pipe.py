import gmsh
import os
from pathlib import Path

# ============================================================
# IMPORTANT USER INPUTS
# ============================================================

OUT_DIR = PARENT_DIR = Path(__file__).resolve().parent / "pipe"
#OUT_DIR = PARENT_DIR = Path(__file__).resolve().parent / "pipe_shark"

# ----------------------------
# Pipe geometry inputs
# ----------------------------

# For outer cylinder, so general pipe geometry
y_base = -47.0 # ~ Half of height
height = 95.0
# For shark:
#y_base = - 37
#height = 74

diameter_i = 21.1 # mm, ID
diameter_o = 24.77 # mm, OD
# For shark:
#diameter_i = 35
#diameter_o = 39 # 2mm thick wall

radius_i = diameter_i / 2.0
radius_o = diameter_o / 2.0

# Inner cylinder trims
TRIM_VALUE = 4.0
y_base_i = y_base + TRIM_VALUE # inner diameter starts higher than outer, so the pipe is closed at the bottom
height_i = height - TRIM_VALUE # then we shorten the height accordingly

# ----------------------------
# Fill geometry inputs
# ----------------------------

# IMPORTANT:
FILL_OVERLAP_SCALAR = 1.05 # Must be > 1.0 to create overlap
diameter_fill = diameter_i * FILL_OVERLAP_SCALAR

if diameter_fill > diameter_o:
    raise ValueError("Fill diameter can't be greater than the outer pipe diameter.")

radius_fill = diameter_fill / 2.0

# IMPORTANT:
# Fill starts slightly below the inner-cylinder base
FILL_VERT_OFFSET = TRIM_VALUE / 2.0
y_base_fill = y_base_i - FILL_VERT_OFFSET

# IMPORTANT:
# Fill height based on inner-cylinder height; this mimics the fact that water wouldn't reach the top of the pipe (if we didn't want spills)
WATER_LEVEL_OFFSET = 5.0
height_fill = height_i - WATER_LEVEL_OFFSET

# ----------------------------
# Mesh density controls
# ----------------------------

# IMPORTANT:
# These are the main inputs controlling mesh fineness on the cylindrical surfaces
N_CIRC = 48   # number of nodes around circular edges
N_Y    = 80   # number of nodes along the axis (y direction)

# Global characteristic lengths, mainly helpful for triangular cases.
# Smaller values generally mean finer mesh

# Coarse
DIV_FACTOR = 7.5

# Med-fine
#DIV_FACTOR = 15.0

# Fine
#DIV_FACTOR = 28.0 # To be comparable to fine in rectangular tank

MESH_SIZE_MIN = min(radius_o, height) / DIV_FACTOR
MESH_SIZE_MAX = min(radius_o, height) / DIV_FACTOR

# If True, use transfinite constraints where possible
# On cylindrical faces this is not always as straightforward as on box faces,
# but circular end curves and axial curves still benefit from explicit control
USE_TRANSFINITE_CURVES = True

# Quad recombination algorithm
RECOMB_ALGO = 1

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reset_model(name):
    gmsh.clear()
    gmsh.model.add(name)

def set_common_options():
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.SaveAll", 1)

    # IMPORTANT:
    # Export directly as legacy VTK
    gmsh.option.setNumber("Mesh.Format", 2)

    gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE_MIN)
    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE_MAX)

    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", RECOMB_ALGO)

def classify_curve_direction(dim, tag, tol=1e-9):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    # Straight axial line: mostly y extent, negligible x/z extent
    if dy > tol and dx < tol and dz < tol:
        return "axial"

    # Circular edge at constant y: negligible y extent
    if dy < tol and (dx > tol or dz > tol):
        return "circumferential"

    return "other"


def apply_transfinite_curve_counts():
    # IMPORTANT:
    # For cylindrical geometries after boolean operations, the exact surface tags
    # are less predictable, but we can still classify curves from their bounding boxes.
    for dim, tag in gmsh.model.getEntities(1):
        ctype = classify_curve_direction(dim, tag)

        if ctype == "axial":
            gmsh.model.mesh.setTransfiniteCurve(tag, N_Y)
        elif ctype == "circumferential":
            gmsh.model.mesh.setTransfiniteCurve(tag, N_CIRC)


def build_pipe_geometry():
    # ============================================================
    # Mirrors gmsh .geo file:
    # Cylinder(1) = {0, y_base + 4, 0, 0, height-4, 0, radius_i};
    # Cylinder(2) = {0, y_base,     0, 0, height,   0, radius_o};
    # BooleanDifference(3) = { Volume{2}; Delete; } { Volume{1}; Delete; };
    # ============================================================

    inner = gmsh.model.occ.addCylinder(0.0, y_base_i, 0.0, 0.0, height_i, 0.0, radius_i)
    outer = gmsh.model.occ.addCylinder(0.0, y_base,   0.0, 0.0, height,   0.0, radius_o)

    # IMPORTANT:
    # Cut the inner cylinder out of the outer cylinder
    # The result is a hollow pipe volume
    out_dimtags, _ = gmsh.model.occ.cut([(3, outer)], [(3, inner)], removeObject=True, removeTool=True)

    gmsh.model.occ.synchronize()

    return out_dimtags


def build_fill_geometry():
    # ============================================================
    # Independent water-fill cylinder
    # ============================================================
    fill = gmsh.model.occ.addCylinder(0.0, y_base_fill, 0.0, 0.0, height_fill, 0.0, radius_fill)
    gmsh.model.occ.synchronize()
    return [(3, fill)]


def set_surface_physical_group_from_volumes(volume_dimtags, name):
    # IMPORTANT:
    # Since we want SURFACE meshes only, we extract all boundary surfaces of the volume
    boundary = gmsh.model.getBoundary(volume_dimtags, oriented=False, recursive=False)
    surf_tags = [tag for dim, tag in boundary if dim == 2]

    if surf_tags:
        pg = gmsh.model.addPhysicalGroup(2, surf_tags)
        gmsh.model.setPhysicalName(2, pg, name)

    return surf_tags


def make_tri_surface_mesh(geometry_builder, filename, physical_name):
    reset_model(physical_name + "_tri")
    set_common_options()

    vols = geometry_builder()
    surf_tags = set_surface_physical_group_from_volumes(vols, physical_name)

    if USE_TRANSFINITE_CURVES:
        apply_transfinite_curve_counts()

    # IMPORTANT:
    # Surface mesh only
    gmsh.model.mesh.generate(2)

    # IMPORTANT:
    # setOrder(1) -> TRI3
    gmsh.model.mesh.setOrder(1)

    gmsh.write(os.path.join(OUT_DIR, filename))



def make_tri6_surface_mesh(geometry_builder, filename, physical_name):
    reset_model(physical_name + "_tri6")
    set_common_options()

    vols = geometry_builder()
    surf_tags = set_surface_physical_group_from_volumes(vols, physical_name)

    if USE_TRANSFINITE_CURVES:
        apply_transfinite_curve_counts()

    gmsh.model.mesh.generate(2)

    # IMPORTANT:
    # Upgrading the same surface topology to quadratic triangles -> TRI6
    gmsh.model.mesh.setOrder(2)

    gmsh.write(os.path.join(OUT_DIR, filename))


def make_quad_surface_mesh(order, second_order_incomplete, geometry_builder, filename, physical_name):
    reset_model(physical_name + "_quad")
    set_common_options()

    vols = geometry_builder()
    surf_tags = set_surface_physical_group_from_volumes(vols, physical_name)

    # IMPORTANT:
    # Recombine all boundary surfaces so triangles are recombined into quads.
    for s in surf_tags:
        gmsh.model.mesh.setRecombine(2, s)

    if USE_TRANSFINITE_CURVES:
        apply_transfinite_curve_counts()

    gmsh.model.mesh.generate(2)

    # IMPORTANT:
    # second_order_incomplete = 1 -> QUAD8
    # second_order_incomplete = 0 -> QUAD9
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", second_order_incomplete)

    # IMPORTANT:
    # order = 1 -> QUAD4
    # order = 2 -> QUAD8 or QUAD9
    gmsh.model.mesh.setOrder(order)

    gmsh.write(os.path.join(OUT_DIR, filename))



def export_all_for_geometry(geometry_builder, prefix, physical_name):
    
    make_tri_surface_mesh(
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
        geometry_builder=build_pipe_geometry,
        prefix="tank_surface",
        physical_name="pipe_surface"
    )

    
    export_all_for_geometry(
        geometry_builder=build_fill_geometry,
        prefix="fill_surface",
        physical_name="pipe_fill_surface"
    )
    
    gmsh.finalize()


if __name__ == "__main__":
    gmsh.initialize()
    main()