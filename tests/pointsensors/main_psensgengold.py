#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import numpy as np
import mooseherder as mh
import pyvale as pyv
import tests.pointsensors.psensconst as psensconst
import psensscalar as pss

def main() -> None:
    print(80*"=")
    print("Gold Output Generator for pyvale Point Sensors")
    print(80*"=")
    print(f"Saving gold output to: {psensconst.GOLD_PATH}\n")

    print("Generating...")


if __name__ == "__main__":
    main()
