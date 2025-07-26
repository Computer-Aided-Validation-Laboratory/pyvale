'''
==============================================================================
TEST: MooseHerd Tests

Authors: Lloyd Fletcher
==============================================================================
'''
import os
import pytest
from pytest import MonkeyPatch
from mooseherder.mooseherd import MooseHerd
from mooseherder.directorymanager import DirectoryManager
import tests.herdchecker as hct


@pytest.fixture
def dir_manager() -> DirectoryManager:
    return DirectoryManager(hct.NUM_DIRS)


@pytest.fixture()
def herd_blank() -> MooseHerd:
    (moose_runner,moose_modifier) = hct.create_moose_objs(hct.MOOSE_INPUT)
    blank_manager = DirectoryManager(hct.NUM_DIRS)
    return MooseHerd([moose_runner],[moose_modifier],blank_manager)


@pytest.fixture()
def herd(dir_manager) -> MooseHerd:
    (moose_runner,moose_modifier) = hct.create_moose_objs(hct.MOOSE_INPUT)
    return MooseHerd([moose_runner],[moose_modifier],dir_manager)


@pytest.fixture()
def herd_gmsh(dir_manager) -> MooseHerd:
    (moose_runner,moose_modifier) = hct.create_moose_objs(hct.MOOSE_INPUT)
    (gmsh_runner,gmsh_modifier) = hct.create_gmsh_objs(hct.GMSH_INPUT)

    runners = [gmsh_runner,moose_runner]
    modifiers = [gmsh_modifier,moose_modifier]

    return MooseHerd(runners,modifiers,dir_manager)


@pytest.fixture(autouse=True)
def setup_teardown(dir_manager):
    # Setup here
    dir_manager.set_base_dir(hct.BASE_DIR)
    dir_manager.create_dirs()
    yield
    # Teardown here
    dir_manager.clear_dirs()


@pytest.fixture()
def moose_sweep() -> list[list[dict]]:
    e_mod = [1e9,2e9]
    p_rat = [0.3,0.35]

    moose_vars = list([])
    for ee in e_mod:
        for pp in p_rat:
            moose_vars.append([{'e_modulus':ee,'p_ratio':pp}])

    return moose_vars


@pytest.fixture()
def moose_sweep_seq() -> list[list[dict]]:
    e_mod = [1e9]
    p_rat = [0.3,0.35]

    moose_vars = list()
    for ee in e_mod:
        for pp in p_rat:
            moose_vars.append([{'e_modulus':ee,'p_ratio':pp}])

    return moose_vars


@pytest.fixture()
def gmsh_sweep() -> list[list[dict | None]]:
    p0 = [1E-3,1.1E-3]
    p1 = [1.5E-3,2E-3]

    gmsh_vars = list()
    for ii in p0:
        for jj in p1:
            gmsh_vars.append([{'p0':ii ,'p1':jj},None])

    return gmsh_vars


@pytest.fixture()
def gmsh_sweep_seq() -> list[list[dict | None]]:
    p0 = [1E-3,]
    p1 = [1.5E-3,2E-3]

    gmsh_vars = list()
    for ii in p0:
        for jj in p1:
            gmsh_vars.append([{'p0':ii ,'p1':jj},None])

    return gmsh_vars


def test_create_herd_blank(herd_blank: MooseHerd) -> None:
    assert herd_blank is not None


def test_create_herd(herd: MooseHerd) -> None:
    assert herd is not None


def test_create_herd_gmsh(herd_gmsh: MooseHerd) -> None:
    assert herd_gmsh is not None


def test_set_input_copy_names(herd: MooseHerd) -> None:
    new_names = ['sim-name']
    herd.set_input_copy_names(new_names)
    assert herd._input_names == new_names

    herd.set_input_copy_names()
    assert herd._input_names == ['sim-1']


def test_set_keep_flag(herd: MooseHerd) -> None:
    herd.set_keep_flag(True)
    assert herd._keep_all == True

    herd.set_keep_flag(False)
    assert herd._keep_all == False

    herd.set_keep_flag()
    assert herd._keep_all == True


@pytest.mark.parametrize(
    ('n_para','expected'),
    (
        (0, 1),
        (-1,1),
        (2.5,2),
        (os.cpu_count()+1,os.cpu_count()) # type: ignore
    )
)
def test_set_num_para_sims(n_para: int, expected: int, herd: MooseHerd):
    herd.set_num_para_sims(n_para)
    assert herd._n_para_sims == expected


@pytest.mark.parametrize(
    ('process','expected'),
    (
        ('MainProcess', '1'),
        ('process-1','1'),
        ('process-2','2'),
        ('process-3','2'),
        ('process-4','1'),
    )
)
def test_get_worker_num(process: str,
                        expected: str,
                        monkeypatch,
                        herd: MooseHerd) -> None:
    monkeypatch.setattr(MooseHerd,'_get_process_name',lambda _: process)
    worker_num = herd._get_worker_num()
    assert worker_num == expected


@pytest.mark.parametrize(
    ('sim_iter','worker_num','keep_all','expected'),
    (
        (0,'1',False, '1'),
        (4,'2',False, '2'),
        (10,'3',False, '3'),
        (0,'1',True, '1'),
        (1,'2',True, '2'),
        (3,'5',True, '4'),
    )
)
def test_get_run_num(sim_iter: int,
                     worker_num: str,
                     keep_all: bool,
                     expected: str,
                     herd: MooseHerd) -> None:
    herd.set_num_para_sims(hct.NUM_PARA)
    herd.set_keep_flag(keep_all)
    run_num = herd._get_run_num(sim_iter,worker_num)
    assert run_num == expected


@pytest.mark.parametrize(
    ('sim_iter','worker_num'),
    (
        (0,'1'),
        (8,'4'),
    )
)
def test_run_once_moose_only(sim_iter: int,
                             worker_num: str,
                             herd: MooseHerd,
                             moose_sweep: list[list[dict]],
                             monkeypatch):
    # Force the process number to be not the main process
    monkeypatch.setattr(MooseHerd, '_get_worker_num', lambda _: worker_num)

    herd.set_num_para_sims(hct.NUM_PARA)

    output_paths = herd.run_once(sim_iter,moose_sweep[0]) # type: ignore

    worker_path = output_paths[0].parent # type: ignore
    stdout_file =  worker_path / 'stdout.processor.0' # type: ignore

    assert output_paths[0].is_file(), 'Output exodus does not exist, MOOSE run failed.' # type: ignore
    assert stdout_file.is_file(), 'stdout file does not exist, MOOSE run failed or redirect flag set incorrectly.'
    assert hct.check_solve_converged(stdout_file), 'MOOSE run did not converge, check stdout file.'
    assert herd._iter_run_time >= 0, 'Iteration run time is less than 0'


@pytest.mark.parametrize(
    ('sim_iter','worker_num'),
    (
        (0,'1'),
        (8,'4'),
    )
)
def test_run_once_with_gmsh(sim_iter: int,
                             worker_num: str,
                             herd_gmsh: MooseHerd,
                             gmsh_sweep: list[list[dict]],
                             monkeypatch):
    # Force the process number to be not the main process
    monkeypatch.setattr(MooseHerd, '_get_worker_num', lambda _: worker_num)

    herd_gmsh.set_num_para_sims(hct.NUM_PARA)

    output_paths = herd_gmsh.run_once(sim_iter,gmsh_sweep[0]) # type: ignore

    worker_path = output_paths[1].parent # type: ignore
    stdout_file =  worker_path / 'stdout.processor.0' # type: ignore

    assert (worker_path / 'gmsh-test.msh').is_file(), 'Gmsh mesh was not created.'
    assert output_paths[1].is_file(), 'Output exodus does not exist, MOOSE run failed.' # type: ignore
    assert stdout_file.is_file(), 'stdout file does not exist, MOOSE run failed or redirect flag set incorrectly.'
    assert hct.check_solve_converged(stdout_file), 'MOOSE run did not converge, check stdout file.'
    assert herd_gmsh._iter_run_time >= 0, 'Iteration run time is less than 0'


@pytest.mark.parametrize(
    ('keep_all', 'expected'),
    (
        (True, 2),
        (False, 1),
    )
)
def test_run_sequential_moose_only(keep_all: bool,
                                   expected: int,
                                   herd: MooseHerd,
                                   dir_manager: DirectoryManager,
                                   moose_sweep_seq: list[list[dict | None]]):
    hct.run_check_seq(keep_all,
                      expected,
                      herd,
                      dir_manager,
                      moose_sweep_seq)

@pytest.mark.parametrize(
    ('keep_all', 'expected'),
    (
        (True, 2),
        (False, 1),
    )
)
def test_run_sequential_with_gmsh(keep_all: bool,
                                   expected: int,
                                   herd_gmsh: MooseHerd,
                                   dir_manager: DirectoryManager,
                                   gmsh_sweep_seq: list[list[dict | None]]):
    hct.run_check_seq(keep_all,
                      expected,
                      herd_gmsh,
                      dir_manager,
                      gmsh_sweep_seq)


@pytest.mark.parametrize(
    ('keep_all',),
    (
        (True,),
        (False,),
    )
)
def test_run_para_moose_only(keep_all: bool,
                            herd: MooseHerd,
                            dir_manager: DirectoryManager,
                            moose_sweep: list[list[dict | None]]):
    hct.run_check_para(keep_all,
                      herd,
                      dir_manager,
                      moose_sweep)

@pytest.mark.parametrize(
    ('keep_all',),
    (
        (True,),
        (False,),
    )
)
def test_run_para_with_gmsh(keep_all: bool,
                            herd_gmsh: MooseHerd,
                            dir_manager: DirectoryManager,
                            gmsh_sweep: list[list[dict | None]]):
    hct.run_check_para(keep_all,
                      herd_gmsh,
                      dir_manager,
                      gmsh_sweep)

