# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Private isolated Gmsh worker; invoked by GmshRunner, never by users."""

import json
import sys
from pathlib import Path

import gmsh


def main(config_path: Path) -> None:
    """Initialise Gmsh, parse a geometry file and write its requested mesh.

    Parameters
    ----------
    config_path : Path
        JSON configuration with input_path, output_path, input_params, mesh_dims,
        mesh_order and msh_version, written by GmshRunner.

    Raises
    ------
    TypeError
        A geometry parameter is not a scalar number, boolean or string.

    Notes
    -----
    Existing meshes in geometry scripts are retained. Otherwise the requested
    mesh dimension is generated. The worker always finalises Gmsh when ordinary
    Python control flow exits its meshing block, including on exceptions.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = ["pyvale-gmsh"]

    for name, value in config["input_params"].items():
        if isinstance(value, (int, float)):
            args.extend(
                (
                    "-setnumber",
                    name,
                    str(int(value) if isinstance(value, bool) else value),
                )
            )
        elif isinstance(value, str):
            args.extend(("-setstring", name, value))
        else:
            raise TypeError("Gmsh geometry input_params must be scalar.")

    gmsh.initialize(args, readConfigFiles=False, interruptible=False)

    try:
        gmsh.open(config["input_path"])

        if not len(gmsh.model.mesh.getNodes()[0]):
            gmsh.model.mesh.generate(config["mesh_dims"])

        if config["mesh_order"] is not None:
            gmsh.model.mesh.setOrder(config["mesh_order"])

        gmsh.option.setNumber("Mesh.MshFileVersion", config["msh_version"])
        gmsh.write(config["output_path"])

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
