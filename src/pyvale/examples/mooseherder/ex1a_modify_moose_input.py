"""
==============================================================================
EXAMPLE: Modify input script for MOOSE with mooseherder

Author: Lloyd Fletcher
==============================================================================
"""
from pathlib import Path
from mooseherder import InputModifier

MOOSE_INPUT = Path("scripts/moose/moose-mech-simple.i")


def main() -> None:
    """main: modify moose input script and write modified file.
    """
    print("-"*80)
    print("EXAMPLE: Modify MOOSE input script")
    print("-"*80)
    moose_mod = InputModifier(MOOSE_INPUT, comment_char="#", end_char="")

    print("Variables found the top of the MOOSE input file:")
    print(moose_mod.get_vars())
    print()

    new_vars = {"n_elem_y": 40, "e_modulus": 3.3e9}
    moose_mod.update_vars(new_vars)

    print("New variables inserted:")
    print(moose_mod.get_vars())
    print()

    moose_save = Path("scripts/moose-mech_mod-vars.i")
    moose_mod.write_file(moose_save)

    print("Modified input script written to:")
    print(moose_save)
    print()

    print("Example complete.")
    print("-"*80)
    print()


if __name__ == "__main__":
    main()
