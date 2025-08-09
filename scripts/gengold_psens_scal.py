#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import pyvale.verif.pointsens as pointsens
import pyvale.verif.pointsensconst as pointsensconst
import pyvale.verif.pointsensscalar as pointsensscalar


def main() -> None:
    tag = "scalar"

    print(80*"=")
    print(f"Gold Output Generator for pyvale {tag} Point Sensors")
    print(80*"=")
    print(f"Saving gold output to: {pointsensconst.GOLD_PATH}\n")

    print(f"Generating 2D gold output for {tag} field point sensors...")
    pointsens.gen_gold_measurements(pointsensscalar.sens_2d_dict())

    print(f"Generating 3D gold output for {tag} field point sensors...")
    pointsens.gen_gold_measurements(pointsensscalar.sens_3d_dict())

if __name__ == "__main__":
    main()
