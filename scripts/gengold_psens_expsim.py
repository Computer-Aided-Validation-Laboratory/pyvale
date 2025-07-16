#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import pyvale.verif.psens as psens
import pyvale.verif.psensconst as psensconst
import pyvale.verif.psensmultiphys as psensmultiphys

def main() -> None:

    print(80*"=")
    print("Gold Output Generator for pyvale Point Sensor Exp. Sim.")
    print(80*"=")
    print(f"Saving gold output to: {psensconst.GOLD_PATH}\n")

    exp_sims_2d = psensmultiphys.exp_sim_2d()

    for ee in exp_sims_2d:
        sensor_arrays = exp_sims_2d[ee].get_sensor_arrays()

        print(80*"-")
        print(f"{ee=}")
        print(f"{len(sensor_arrays)=}")
        print(80*"-")

        exp_data = exp_sims_2d[ee].run_experiments()
        exp_stats = exp_sims_2d[ee].calc_stats()






if __name__ == "__main__":
    main()
