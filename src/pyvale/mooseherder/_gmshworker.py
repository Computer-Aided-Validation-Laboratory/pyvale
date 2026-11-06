"""Private isolated Gmsh worker; invoked by GmshRunner, never by users."""

import json
import sys
from pathlib import Path

import gmsh


def main(config_path: Path) -> None:
    """Parse geometry, generate missing mesh and write the declared output."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = ["pyvale-gmsh"]
    for name, value in config["parameters"].items():
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
            raise TypeError("Gmsh geometry parameters must be scalar.")
    gmsh.initialize(args, readConfigFiles=False, interruptible=False)
    try:
        gmsh.open(config["input_path"])
        # Legacy scripts may already contain Mesh and Save commands.
        if not len(gmsh.model.mesh.getNodes()[0]):
            gmsh.model.mesh.generate(config["dimension"])
        if config["order"] is not None:
            gmsh.model.mesh.setOrder(config["order"])
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(config["output_path"])
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
