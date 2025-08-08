#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import pyvale.verif.psens as psens
import pyvale.verif.psensconst as psensconst
import pyvale.verif.psensvector as psensvector


def main() -> None:
    tag = "vector"
    print(80*"=")
    print(f"Gold Output Generator for pyvale {tag} Point Sensors")
    print(80*"=")
    print(f"Saving gold output to: {psensconst.GOLD_PATH}\n")

    print(f"Generating 2D gold output for {tag} field point sensors...")
    psens.gen_gold_measurements(psensvector.sens_2d_dict())

    print(f"Generating 3D gold output for {tag} field point sensors...")
    psens.gen_gold_measurements(psensvector.sens_3d_dict())

if __name__ == "__main__":
    main()
