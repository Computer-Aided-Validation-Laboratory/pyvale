#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

from pathlib import Path
import pyvale.sensorsim as sens
import pyvale.dataset as dataset
import pyvale.mooseherder as mh
import pyvale.verif.matchsimdata as verif


def main() -> None:
    data_path = dataset.element_case_output_path(dataset.EElemTest.HEX20)
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()

    gold_path = Path.cwd()/"tests"/"mooseherder"/"txt_gold"

    load_opts = mh.SimTxtLoadOpts()
    save_opts = mh.SimDataSaveOpts(sim_tag="hex20")

    suffix = ".npy"

    coord_path = gold_path / (save_opts.get_coord_name() + suffix)
    time_path = gold_path / (save_opts.get_time_name() + suffix)

    field_slices = {"disp_x": slice(0,1),
                    "disp_y": slice(1,2),
                    "disp_z": slice(2,3),
                    "strain_xx": slice(3,4),
                    "strain_xy": slice(4,5),
                    "strain_xz": slice(5,6),
                    "strain_yy": slice(6,7),
                    "strain_yz": slice(7,8),
                    "strain_zz": slice(8,9),
                    "temperature": slice(9,10),}

    field_pattern = f"hex20_node_field_frame*{suffix}"

    sim_loader = mh.SimTxtLoader(files_path=gold_path,
                                 coords=coord_path,
                                 time_steps=time_path,
                                 node_file_pattern=field_pattern,
                                 node_slices=field_slices,
                                 glob_file=None,
                                 glob_slices=None,
                                 load_opts=load_opts)

    sim_data_load = sim_loader.load_all_sim_data()

    sens.print_sim_data(sim_data_load)

    sim_data.connect = None
    sim_data.glob_vars = None
    match = verif.match_sim_data(sim_data,sim_data_load)


    print(80*"=")
    for mm in match:
        print(f"{mm}={match[mm]}")
    print(80*"=")

    fails = verif.match_sim_data_get_fails(sim_data,sim_data_load)
    print(f"{fails=}")
    print()



if __name__ == "__main__":
    main()