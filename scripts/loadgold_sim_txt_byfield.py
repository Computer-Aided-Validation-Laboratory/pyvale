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
    data_path: Path = dataset.element_case_output_path(dataset.EElemTest.HEX20)
    sim_data: mh.SimData = mh.ExodusLoader(data_path).load_all_sim_data()

    project_root: Path = Path(__file__).resolve().parents[1]
    gold_path: Path = project_root/"tests"/"mooseherder"/"txt_gold"

    load_opts = mh.SimLoadOpts(node_field_header=None)
    save_opts = mh.SimDataSaveOpts(sim_tag="hex20")

    suffix = ".npy"
    coords_file = save_opts.get_coord_name() + suffix
    time_step_file = save_opts.get_time_name() + suffix

    prefix = "hex20"
    field_keys = {"disp_x",
                  "disp_y",
                  "disp_z",
                  "strain_xx",
                  "strain_xy",
                  "strain_xz",
                  "strain_yy",
                  "strain_yz",
                  "strain_zz",
                  "temperature"}

    field_prefix = f"{prefix}_node_field"

    field_patterns = {}
    for ff in field_keys:
        field_patterns[ff] = f"{field_prefix}_{ff}{suffix}"

    for ff in field_patterns:
        print(f"{ff}: {field_patterns[ff]}")
        
    print()

    connect_pattern: str = f"{prefix}_connect*{suffix}"

    glob_file: str = f"{prefix}_glob{suffix}"
    glob_slices = {"disp_x_max":slice(0,1),
                   "disp_y_max":slice(1,2),
                   "disp_z_max":slice(2,3),
                   "react_y_bot":slice(3,4),
                   "react_y_top":slice(4,5),}

    sim_loader = mh.SimLoaderByField(load_dir=gold_path,
                                     coords_file=coords_file,
                                     time_step_file=time_step_file,
                                     node_field_files=field_patterns,
                                     connect_files=connect_pattern,   
                                     glob_file=glob_file,
                                     glob_slices=glob_slices,
                                     load_opts=load_opts)

    sim_data_load: mh.SimData = sim_loader.load_all_sim_data()

    sens.print_sim_data(sim_data_load)

    match_check = verif.match_sim_data(sim_data,sim_data_load)

    print(80*"=")
    for mm in match_check:
        print(f"{mm}={match_check[mm]}")
    print(80*"=")

    fails = verif.match_sim_data_get_fails(sim_data,sim_data_load)
    print(f"{fails=}")


if __name__ == "__main__":
    main()
