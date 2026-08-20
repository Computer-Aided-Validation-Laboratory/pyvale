#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

from pathlib import Path
import shutil
import pytest
import pyvale.mooseherder as mh
import pyvale.dataio as io
import pyvale.data as dataset
import pyvale.verif.matchsimdata as verif

TXT_GOLD_PATH: Path = Path(__file__).resolve().parent / "txt_gold"
TEMP_OUTPUT_PATH: Path = Path(__file__).resolve().parent / "temp" 

@pytest.fixture(autouse=True)
def setup_teardown():

    output_path = TEMP_OUTPUT_PATH
    if not output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

    yield

    shutil.rmtree(output_path)


@pytest.fixture()
def sim_data() -> io.SimData:
    data_path = dataset.element_case_output_path(dataset.EElemTest.HEX20)
    return mh.ExodusLoader(data_path).load_all_sim_data()


def test_save_sim_data(sim_data: io.SimData) -> None:
    save_opts = io.SimDataSaveOpts(io.ESaveFieldOpt.BOTH,
                                   io.ESaveArray.BOTH,
                                   sim_tag="hex20")

    io.save_sim_data_to_arrays(TEMP_OUTPUT_PATH,
                               sim_data,
                               save_opts)

    # Check against gold directory contents to make sure everything is there
    save_dir = TEMP_OUTPUT_PATH
    files_save = {(ff.name,ff.suffix) for ff in save_dir.iterdir() if ff.is_file()}
    gold_dir = TXT_GOLD_PATH
    files_gold = {(ff.name,ff.suffix) for ff in gold_dir.iterdir() if ff.is_file()}

    assert files_save == files_gold, ("Saved file do not match those in gold "
                                      + f"dir:\n    {files_save=}.")


@pytest.mark.parametrize(
    ("suffix",),
    (
        (".csv",),
        (".npy",),
    )
)
def test_load_sim_data_by_time(sim_data: io.SimData,
                                suffix: str) -> None:

    load_opts = io.SimLoadOpts()

    save_opts = io.SimDataSaveOpts(sim_tag="hex20")
    coord_path = TXT_GOLD_PATH / (save_opts.get_coord_name() + suffix)
    time_path = TXT_GOLD_PATH / (save_opts.get_time_name() + suffix)

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
    # NOTE: known issue with connect dictionary having prefix on keys where gold
    # does not
    connect_pattern = f"hex20_connect*"
    
    sim_loader = io.SimLoaderByTime(load_dir=TXT_GOLD_PATH,
                                    coords_file=coord_path,
                                    time_step_file=time_path,
                                    node_files=field_pattern,
                                    node_slices=field_slices,
                                    connect_files=None,
                                    glob_file=None,
                                    glob_slices=None,
                                    load_opts=load_opts)

    sim_data_load = sim_loader.load_all_sim_data()

    sim_data.connect = None
    sim_data.glob_vars = None
    fails = verif.match_sim_data_get_fails(sim_data,sim_data_load)

    assert not fails, "\n".join(fails)


@pytest.mark.parametrize(
    ("suffix",),
    (
        (".csv",),
        (".npy",),
    )
)
def test_save_load_sim_data_by_frame(sim_data: io.SimData, suffix: str) -> None:

    save_opts = io.SimDataSaveOpts(io.ESaveFieldOpt.BOTH,
                                   io.ESaveArray.BOTH,
                                   sim_tag="hex20")

    io.save_sim_data_to_arrays(TEMP_OUTPUT_PATH,
                               sim_data,
                               save_opts)

    # Check against gold directory contents to make sure everything is there
    save_dir = TEMP_OUTPUT_PATH
    files_save = {(ff.name,ff.suffix) for ff in save_dir.iterdir() if ff.is_file()}
    gold_dir = TXT_GOLD_PATH
    files_gold = {(ff.name,ff.suffix) for ff in gold_dir.iterdir() if ff.is_file()}

    assert files_save == files_gold, ("Saved file do not match those in gold "
                                      + f"dir:\n    {files_save=}.")


    load_opts = io.SimLoadOpts()

    save_opts = io.SimDataSaveOpts(sim_tag="hex20")
    coord_path = TEMP_OUTPUT_PATH / (save_opts.get_coord_name() + suffix)
    time_path = TEMP_OUTPUT_PATH / (save_opts.get_time_name() + suffix)

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

    # NOTE: known issue with connect dictionary having prefix on keys where gold
    # does not
    connect_pattern = f"hex20_connect*"
    
    sim_loader = io.SimLoaderByTime(load_dir=TXT_GOLD_PATH,
                                    coords_file=coord_path,
                                    time_step_file=time_path,
                                    node_files=field_pattern,
                                    node_slices=field_slices,
                                    connect_files=None,
                                    glob_file=None,
                                    glob_slices=None,
                                    load_opts=load_opts)
                                    
    sim_data_load = sim_loader.load_all_sim_data()

    sim_data.connect = None
    sim_data.glob_vars = None
    fails = verif.match_sim_data_get_fails(sim_data,sim_data_load)

    assert not fails, "\n".join(fails)


@pytest.mark.parametrize(
    ("suffix",),
    (
        (".csv",),
        (".npy",),
    )
)
def test_load_sim_data_by_field(sim_data: io.SimData, suffix: str) -> None:

    load_opts = io.SimLoadOpts(node_field_header=None)

    save_opts = io.SimDataSaveOpts(sim_tag="hex20")
    coord_path = TXT_GOLD_PATH / (save_opts.get_coord_name() + suffix)
    time_path = TXT_GOLD_PATH / (save_opts.get_time_name() + suffix)

    prefix = "hex20_node_field"

    field_keys = {"disp_x","disp_y","disp_z","strain_xx","strain_xy","strain_xz"
                 ,"strain_yy","strain_yz","strain_zz","temperature"}

    field_patterns = {}
    for ff in field_keys:
        field_patterns[ff] = f"{prefix}_{ff}{suffix}"

    # NOTE: known issue with connect dictionary having prefix on keys where gold
    # does not
    connect_pattern = f"hex20_connect*"

    sim_loader = io.SimLoaderByField(load_dir=TXT_GOLD_PATH,
                                     coords_file=coord_path,
                                     time_step_file=time_path,
                                     node_field_files=field_patterns,
                                     connect_files=None,
                                     glob_file=None,
                                     glob_slices=None,
                                     load_opts=load_opts)

    sim_data_load = sim_loader.load_all_sim_data()

    sim_data.connect = None
    sim_data.glob_vars = None
    fails = verif.match_sim_data_get_fails(sim_data,sim_data_load)

    assert not fails, "\n".join(fails)


@pytest.mark.parametrize(
    ("suffix",),
    (
        (".csv",),
        (".npy",),
    )
)
def test_save_load_sim_data_by_field(sim_data: io.SimData, suffix: str) -> None:

    save_opts = io.SimDataSaveOpts(io.ESaveFieldOpt.BOTH,
                                   io.ESaveArray.BOTH,
                                   sim_tag="hex20")

    io.save_sim_data_to_arrays(TEMP_OUTPUT_PATH,
                               sim_data,
                               save_opts)

    # Check against gold directory contents to make sure everything is there
    save_dir = TEMP_OUTPUT_PATH
    files_save = {(ff.name,ff.suffix) for ff in save_dir.iterdir() if ff.is_file()}
    gold_dir = TXT_GOLD_PATH
    files_gold = {(ff.name,ff.suffix) for ff in gold_dir.iterdir() if ff.is_file()}

    assert files_save == files_gold, ("Saved file do not match those in gold "
                                      + f"dir:\n    {files_save=}.")


    load_opts = io.SimLoadOpts(node_field_header=None)

    save_opts = io.SimDataSaveOpts(sim_tag="hex20")
    coord_path = TEMP_OUTPUT_PATH / (save_opts.get_coord_name() + suffix)
    time_path = TEMP_OUTPUT_PATH / (save_opts.get_time_name() + suffix)


    prefix = "hex20_node_field"

    field_keys = {"disp_x","disp_y","disp_z","strain_xx","strain_xy","strain_xz"
                 ,"strain_yy","strain_yz","strain_zz","temperature"}

    field_patterns = {}
    for ff in field_keys:
        field_patterns[ff] = f"{prefix}_{ff}{suffix}"

    # NOTE: known issue with connect dictionary having prefix on keys where gold
    # does not
    connect_pattern = f"hex20_connect*"

    sim_loader = io.SimLoaderByField(load_dir=TXT_GOLD_PATH,
                                     coords_file=coord_path,
                                     time_step_file=time_path,
                                     node_field_files=field_patterns,
                                     connect_files=None,
                                     glob_file=None,
                                     glob_slices=None,
                                     load_opts=load_opts)

    sim_data_load = sim_loader.load_all_sim_data()

    sim_data.connect = None
    sim_data.glob_vars = None
    fails = verif.match_sim_data_get_fails(sim_data,sim_data_load)

    assert not fails, "\n".join(fails)

