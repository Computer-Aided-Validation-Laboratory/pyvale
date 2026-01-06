#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import pyvale.verif.pointsens as pointsens
import pyvale.verif.pointsensconst as pointsensconst
import pyvale.verif.pointsensscalar as pointsensscalar
import pyvale.verif.pointsensvector as pointsensvector
import pyvale.verif.pointsenstensor as pointsenstensor


def main() -> None:

    print(80*"=")
    print("Gold Output Generator for pyvale Point Sensors")
    print(80*"=")
    print(f"Saving gold output to: {pointsensconst.GOLD_PATH}\n")

    sens = [pointsensscalar.sens_arrays_2d_dict(),
            pointsensscalar.sens_arrays_3d_dict(),
            pointsensvector.sens_arrays_2d_dict(),
            pointsensvector.sens_arrays_3d_dict(),
            pointsenstensor.sens_arrays_2d_dict(),
            pointsenstensor.sens_arrays_3d_dict(),]

    for ss in sens:
        pointsens.gen_gold_measurements(ss)

    print(80*"-")
    print("Gold output generation complete.\n")

if __name__ == "__main__":
    main()
