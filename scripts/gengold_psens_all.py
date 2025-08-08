#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import pyvale.verif.psens as psens
import pyvale.verif.psensconst as psensconst
import pyvale.verif.psensscalar as psensscalar
import pyvale.verif.psensvector as psensvector
import pyvale.verif.psenstensor as psenstensor


def main() -> None:

    print(80*"=")
    print("Gold Output Generator for pyvale Point Sensors")
    print(80*"=")
    print(f"Saving gold output to: {psensconst.GOLD_PATH}\n")

    sens = [psensscalar.sens_2d_dict(),
            psensscalar.sens_3d_dict(),
            psensvector.sens_2d_dict(),
            psensvector.sens_3d_dict(),
            psenstensor.sens_2d_dict(),
            psenstensor.sens_3d_dict(),]

    for ss in sens:
        psens.gen_gold_measurements(ss)

    print(80*"-")
    print("Gold output generation complete.\n")

if __name__ == "__main__":
    main()
