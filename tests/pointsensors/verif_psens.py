#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import numpy as np

import pyvale.verif.psensscalar as psensscalar

def main() -> None:
    sens_dict = psensscalar.sens_2d_dict()

    for ss in sens_dict:
        calc1 = sens_dict[ss].calc_measurements()
        errs1 = sens_dict[ss].get_errors_total()
        calc2 = sens_dict[ss].calc_measurements()
        errs2 = sens_dict[ss].get_errors_total()


        print(80*"=")
        print(ss)
        print(sens_dict[ss]._error_integrator)
        print()
        print(calc1[0,0,-6:-1])
        print(calc2[0,0,-6:-1])
        print()
        if errs1 is not None and errs2 is not None:
            print(errs1[0,0,-6:-1])
            print(errs2[0,0,-6:-1])
        print(80*"=")


if __name__ == "__main__":
    main()