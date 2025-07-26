'''
==============================================================================
HerdChecker: Helper functions for testing MooseHerder

Authors: Lloyd Fletcher
==============================================================================
'''

import os
from typing import Any
from pathlib import Path
from pprint import pprint
from mooseherder.inputmodifier import InputModifier
from mooseherder.mooserunner import MooseRunner
from mooseherder.mooseconfig import MooseConfig
from mooseherder.gmshrunner import GmshRunner
from mooseherder.mooseherd import MooseHerd
from mooseherder.directorymanager import DirectoryManager


NUM_DIRS = 4
NUM_PARA = 4
NUM_CALLS = 3
BASE_DIR = Path('tests/')

MOOSE_PATH = Path.home()/'moose'
MOOSE_APP_PATH = Path().home()/'proteus'
MOOSE_APP_NAME = 'proteus-opt'
MOOSE_INPUT = Path('tests/moose/moose-test.i')

GMSH_APP_PATH = Path().home() / 'gmsh/bin/gmsh'
GMSH_INPUT = Path('tests/gmsh/gmsh-test.geo')

OUTPUT_PATH = Path('tests/output/')


def create_moose_config() -> MooseConfig:
    return MooseConfig({'main_path': MOOSE_PATH,
                        'app_path': MOOSE_APP_PATH,
                        'app_name': MOOSE_APP_NAME})


def create_moose_objs(input_file: Path) -> tuple[MooseRunner,InputModifier]:
    moose_config = create_moose_config()
    moose_runner = MooseRunner(moose_config)

    moose_input = input_file
    moose_modifier = InputModifier(moose_input,'#','')

    return (moose_runner,moose_modifier)


def create_gmsh_objs(input_file: Path) -> tuple[GmshRunner,InputModifier]:
    gmsh_input = input_file
    gmsh_modifier = InputModifier(gmsh_input,'//',';')
    gmsh_runner = GmshRunner(GMSH_APP_PATH)
    gmsh_runner.set_input_file(gmsh_input)

    return (gmsh_runner,gmsh_modifier)

def check_solve(check_file: Path, check_str: str) -> int:
    check_count = 0
    if check_file.is_file():
        with open(check_file,'r', encoding="utf-8") as so:
            stdout_lines = so.readlines()
            for ll in stdout_lines:
                if check_str in ll:
                    check_count += 1
    return check_count


def check_solve_converged(check_stdout: Path) -> int:
    return check_solve(check_stdout,'Solve Converged!')


def check_solve_error(check_stdout: Path) -> int:
    return check_solve(check_stdout,'*** ERROR ***')


def check_output_key_file_count(check_dir: Path) -> int:
    work_dir_files = os.listdir(check_dir)

    key_count = 0
    for ff in work_dir_files:
        if 'output-key' in ff:
            key_count += 1

    return key_count


def check_run_sweep(check_herd: MooseHerd,
                    dir_manager: DirectoryManager,
                    run_call: int) -> None:
    for ll in dir_manager.get_output_paths():
        for ff in ll:
            if ff is not None:
                assert ff.is_file(), \
                    f"Simulation output {ff} does not exist."

    assert check_herd.get_sweep_time() >= 0, \
        'Sweep run time is less than 0.'
    assert dir_manager.get_output_key_file(check_herd.get_sweep_iter()), \
        'Output key file was not written.'

    output_key_count = check_output_key_file_count(
        dir_manager.get_output_key_file(
            check_herd.get_sweep_iter()).parent)
    assert output_key_count == run_call


def check_input_output_count_sequential(check_herd: MooseHerd,
                                        dir_manager: DirectoryManager,
                                        run_call: int) -> None:
    (input_count, output_count) = get_input_output_count(
        dir_manager.get_all_run_dirs(),'.i','_out.e')

    if not check_herd._keep_all:
        num_sims = 1
    else:
        num_sims = run_call*len(check_herd._var_sweep)

    assert input_count == num_sims
    assert output_count == num_sims


def check_input_output_count_para(check_herd: MooseHerd,
                                  dir_manager: DirectoryManager,
                                  run_call: int) -> None:
    (input_count, output_count) = get_input_output_count(
        dir_manager.get_all_run_dirs(),'.i','_out.e')

    if not check_herd._keep_all:
        num_sims = len(check_herd._var_sweep)
    else:
        num_sims = run_call*len(check_herd._var_sweep)

    assert input_count == num_sims
    assert output_count == num_sims


def get_input_output_count(run_dirs: list[Path],
                           in_suffix: str,
                           out_suffix: str) -> tuple[int,int]:
    # Go through all work directories and count the inputs/exoduses
    input_count = 0
    output_count = 0

    for rr in run_dirs:
        dir_files = os.listdir(rr)
        for ff in dir_files:
            if out_suffix in ff:
                output_count += 1
            elif in_suffix in ff:
                input_count += 1

    return (input_count, output_count)


def run_check_seq(keep_all: bool,
                  expected: int,
                  herd: MooseHerd,
                  dir_manager: DirectoryManager,
                  sweep_vars: list[list[dict | None]]) -> None:

    herd.set_keep_flag(keep_all)
    herd.set_num_para_sims(NUM_PARA)

    herd.run_sequential(sweep_vars)
    check_run_sweep(herd, dir_manager, run_call = 1)
    check_input_output_count_sequential(herd, dir_manager, run_call = 1)

    herd.run_sequential(sweep_vars)
    check_run_sweep(herd, dir_manager, run_call = expected)
    check_input_output_count_sequential(herd, dir_manager, run_call = expected)


def run_check_para(keep_all: bool,
                   herd: MooseHerd,
                   dir_manager: DirectoryManager,
                   sweep_vars: list[list[dict | None]]) -> None:
    herd.set_keep_flag(keep_all)
    herd.set_num_para_sims(NUM_PARA)

    for rr in range(NUM_CALLS):
        if keep_all:
            run_check = rr+1
        else:
            run_check = 1

        herd.run_para(sweep_vars)
        check_run_sweep(herd, dir_manager, run_check)
        check_input_output_count_para(herd, dir_manager, run_check)

def debug_print(to_print: Any) -> None:
    print()
    print('-'*80)
    pprint(to_print)
    print('-'*80)
    print()

