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

    sim_list = psensmultiphys.simdata_list_2d()
    for ss in sim_list:
        print(f"{ss.time.shape=}")
    print()

    exp_sims_2d = psensmultiphys.exp_sim_2d()

    for ii,ee in enumerate(exp_sims_2d):
        exp_data = exp_sims_2d[ee].run_experiments()
        exp_stats = exp_sims_2d[ee].calc_stats()
        print(80*"=")
        print(f"{type(exp_data[0])=}")
        print(f"{type(exp_stats[0])=}")
        print(80*"=")






if __name__ == "__main__":
    main()
