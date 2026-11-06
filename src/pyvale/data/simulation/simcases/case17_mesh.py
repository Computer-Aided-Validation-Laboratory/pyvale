"""Python Gmsh mesh for case17: a 100 x 150 mm plate with a 12.5 mm-radius hole.

This is an alternative mesh for the existing case17.i weak-plane-stress model.
It preserves the plate geometry, mm units, and bc-base/bc-top boundary names;
the .geo version uses mapped quadrilaterals, whereas this script uses triangles.
GmshRunner supplies --config with parameters and the requested output_path.
"""

import argparse
import json
import math
from pathlib import Path

import gmsh


def main() -> None:
    """Generate a headless mesh compatible with the packaged case17.i deck."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    mesh_size = float(config["parameters"].get("mesh_size_mm", 15.0))
    if not math.isfinite(mesh_size) or mesh_size <= 0:
        raise ValueError("mesh_size_mm must be finite and positive.")
    if config["dimension"] != 2:
        raise ValueError("case17 is a 2D plate.")
    order = config["order"] if config["order"] is not None else 2
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.model.add("case17")
        plate = gmsh.model.occ.addRectangle(0, 0, 0, 100, 150)
        hole = gmsh.model.occ.addDisk(50, 75, 0, 12.5, 12.5)
        surfaces, _ = gmsh.model.occ.cut([(2, plate)], [(2, hole)])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        tolerance = 1e-6  # mm; include the CAD kernel's bounding-box tolerance.
        for tag, (name, y) in enumerate((("bc-base", 0), ("bc-top", 150)), 1):
            edges = gmsh.model.getEntitiesInBoundingBox(
                -tolerance,
                y - tolerance,
                -tolerance,
                100 + tolerance,
                y + tolerance,
                tolerance,
                1,
            )
            if not edges:
                raise RuntimeError(f"Could not find case17 boundary {name}.")
            gmsh.model.addPhysicalGroup(1, [edge[1] for edge in edges], tag, name)
        gmsh.model.addPhysicalGroup(2, [surface[1] for surface in surfaces], 3, "plate")
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(config["output_path"])
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
