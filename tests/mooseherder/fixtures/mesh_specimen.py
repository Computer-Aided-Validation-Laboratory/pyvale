"""A user-supplied mesh script implementing GmshRunner's JSON CLI contract."""

import argparse
import json
from pathlib import Path

import gmsh


def main() -> None:
    """Generate a transfinite unit-square mesh with named MOOSE boundaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    n = config["parameters"].get("n", 3)
    if type(n) is not int or n < 2:
        raise ValueError("n must be an integer >= 2.")
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.model.add("specimen")
        geo = gmsh.model.geo
        for tag, (x, y) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1)), 1):
            geo.addPoint(x, y, 0, 1, tag)
        for tag in range(1, 5):
            geo.addLine(tag, tag % 4 + 1, tag)
        geo.addCurveLoop([1, 2, 3, 4], 1)
        geo.addPlaneSurface([1], 1)
        geo.synchronize()
        for tag in range(1, 5):
            gmsh.model.mesh.setTransfiniteCurve(tag, n)
        gmsh.model.mesh.setTransfiniteSurface(1)
        gmsh.model.mesh.setRecombine(2, 1)
        for tag, name in enumerate(("bottom", "right", "top", "left"), 1):
            gmsh.model.addPhysicalGroup(1, [tag], tag, name)
        gmsh.model.addPhysicalGroup(2, [1], 5, "specimen")
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(config["output_path"])
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
