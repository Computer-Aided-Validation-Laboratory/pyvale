"""
==============================================================================
EXAMPLE: Modify input script for gmsh with mooseherder

Author: Lloyd Fletcher
==============================================================================
"""
from pathlib import Path
from mooseherder import InputModifier

GMSH_INPUT = Path("scripts/gmsh/gmsh_tens_spline_2d.geo")


def main() -> None:
    """main: modify gmsh input and save to file
    """
    print("-"*80)
    print("EXAMPLE: Modify gmsh input script")
    print("-"*80)
    gmsh_mod = InputModifier(GMSH_INPUT, "//", ";")

    print("Variables found the top of the gmsh input file:")
    print(gmsh_mod.get_vars())
    print()

    new_vars = {"p0": 0.0018, "p1": 0.0012, "p2": 0.001}
    gmsh_mod.update_vars(new_vars)

    print("New variables inserted:")
    print(gmsh_mod.get_vars())
    print()

    gmsh_save = Path("scripts/gmsh_tens_spline_2d-mod_vars.geo")
    gmsh_mod.write_file(gmsh_save)

    print("Modified input script written to:")
    print(gmsh_save)
    print()

    print("Example complete.")
    print("-"*80)
    print()


if __name__ == "__main__":
    main()
