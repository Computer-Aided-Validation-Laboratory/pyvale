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

    project_root = Path(__file__).resolve().parents[1]
    gold_path = project_root/"tests"/"mooseherder"/"txt_gold"

    load_opts = mh.SimTxtLoadOpts(node_field_header=None)
    save_opts = mh.SimDataSaveOpts(sim_tag="hex20")

    suffix = ".npy"

    coord_path = gold_path / (save_opts.get_coord_name() + suffix)
    time_path = gold_path / (save_opts.get_time_name() + suffix)

    field_slices = {"disp_x": slice(None),
                    "disp_y": slice(None),
                    "disp_z": slice(None),
                    "strain_xx": slice(None),
                    "strain_xy": slice(None),
                    "strain_xz": slice(None),
                    "strain_yy": slice(None),
                    "strain_yz": slice(None),
                    "strain_zz": slice(None),
                    "temperature": slice(None),}

    prefix = "hex20_node_field"

    field_patterns = {}
    for ff in field_slices:
        field_patterns[ff] = f"{prefix}_{ff}{suffix}"

    for ff in field_patterns:
        print(f"{ff}: {field_patterns[ff]}")

    print()

    sim_loader = mh.SimTxtLoader(files_path=gold_path,
                                 coords=coord_path,
                                 time_steps=time_path,
                                 node_file_pattern=field_patterns,
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


if __name__ == "__main__":
    main()