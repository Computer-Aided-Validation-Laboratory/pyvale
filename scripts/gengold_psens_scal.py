#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import pyvale.verif.psensconst as psensconst
import pyvale.verif.psensscalar as psensscalar


def main() -> None:
    print(80*"=")
    print("Gold Output Generator for pyvale Point Sensors")
    print(80*"=")
    print(f"Saving gold output to: {psensconst.GOLD_PATH}\n")

    print("Generating 2D gold output for scalar field point sensors...")
    psensscalar.gen_gold_2d()


if __name__ == "__main__":
    main()
