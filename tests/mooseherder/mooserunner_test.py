'''
==============================================================================
TEST: MooseRunner

Authors: Lloyd Fletcher
==============================================================================
'''

import os
from pathlib import Path
import pytest
from mooseherder.mooserunner import MooseRunner
import tests.herdchecker as hc


@pytest.fixture()
def runner() -> MooseRunner:
    moose_config = hc.create_moose_config()
    return MooseRunner(moose_config)

@pytest.fixture()
def input_path() -> Path:
    return Path('tests/moose/moose-test.i')

@pytest.fixture()
def input_noexist() -> Path:
    return Path('tests/moose/moose-test-noexist.i')

@pytest.fixture()
def input_broken() -> Path:
    return Path('tests/moose/moose-test-broken.i')

@pytest.fixture()
def input_runner(input_path: Path) -> MooseRunner:
    moose_config = hc.create_moose_config()
    my_runner = MooseRunner(moose_config)
    my_runner.set_input_file(input_path)
    return my_runner

@pytest.fixture(autouse=True)
def setup_teardown(input_runner: MooseRunner):
    # Setup here
    yield
    # Teardown here - remove output exodus files
    moose_files = os.listdir(input_runner.get_input_dir())
    for ff in moose_files:
        if '.e' in ff:
            os.remove(input_runner.get_input_dir() / ff) # type: ignore

    stdout_files = os.listdir(input_runner.get_input_dir())
    for ff in stdout_files:
        if 'stdout.processor' in ff:
            os.remove(input_runner.get_input_dir() / ff) # type: ignore

def test_set_env_vars(runner: MooseRunner) -> None:
    runner.set_env_vars()
    assert os.environ['CC'] == 'mpicc'
    assert os.environ['CXX'] == 'mpicxx'
    assert os.environ['F90'] == 'mpif90'
    assert os.environ['F77'] == 'mpif77'
    assert os.environ['FC'] == 'mpif90'
    assert os.environ['MOOSE_DIR'] == str(runner._config['main_path'])

@pytest.mark.parametrize(
    ('n_threads','expected'),
    (
        (0, 1),
        (-1,1),
        (2.5,2),
        (os.cpu_count()+1,os.cpu_count()) # type: ignore
    )
)
def test_set_threads(n_threads: int,expected: int ,runner: MooseRunner) -> None:
    runner.set_threads(n_threads)
    assert runner._n_threads == expected

@pytest.mark.parametrize(
    ('n_tasks','expected'),
    (
        (0, 1),
        (-1,1),
        (2.5,2),
        (os.cpu_count()+1,os.cpu_count()) # type: ignore
    )
)
def test_set_tasks(n_tasks: int, expected: int, runner: MooseRunner):
    runner.set_tasks(n_tasks)
    assert runner._n_tasks == expected


@pytest.mark.parametrize(
    ('in_flag','expected'),
    (
        (True,True),
        (False,False)
    )
)
def test_set_stdout(in_flag: bool, expected: bool, runner: MooseRunner):
    runner.set_stdout(in_flag)
    assert runner._redirect_stdout == expected


def test_set_input_file(runner: MooseRunner, input_path: Path) -> None:
    runner.set_input_file(input_path)

    assert runner._input_path == input_path


def test_set_input_file_err(runner: MooseRunner) -> None:
    with pytest.raises(FileNotFoundError) as err_info:
        new_input = Path('tests/moose/moose-test-noexist.i')
        runner.set_input_file(new_input)

    msg, = err_info.value.args
    assert msg == 'Input file does not exist.'


def test_get_input_strs(runner: MooseRunner, input_runner: MooseRunner) -> None:
    assert runner.get_input_dir() is None
    assert runner.get_input_tag() == ''
    assert input_runner.get_input_dir() == Path('tests/moose/')
    assert input_runner.get_input_tag() == 'moose-test'


def test_get_output_path(runner: MooseRunner,input_runner: MooseRunner):
    assert runner.get_output_path() is None
    assert input_runner.get_output_path() == Path('tests/moose/moose-test_out.e')


@pytest.mark.parametrize(
    ('opts','expected'),
    (
        ((1,1,False), ['proteus-opt','--n-threads=1','-i','moose-test.i']),
        ((1,2,False), ['proteus-opt','--n-threads=2','-i','moose-test.i']),
        ((1,2,True), ['proteus-opt','--n-threads=2','-i','moose-test.i','--redirect-stdout']),
        ((2,2,False), ['mpirun','-np','2','proteus-opt','--n-threads=2','-i','moose-test.i']),
        ((2,2,True), ['mpirun','-np','2','proteus-opt','--n-threads=2','-i','moose-test.i','--redirect-stdout']),
    )
)
def test_assemble_arg_list(opts: tuple[int,int,bool],
                          expected: str,
                          input_runner: MooseRunner) -> None:
    input_runner.set_run_opts(opts[0],opts[1],opts[2])
    assert input_runner.assemble_arg_list() == expected


@pytest.mark.parametrize(
    ('opts','expected'),
    (
        ((1,1,False), ['proteus-opt','--n-threads=1','-i','moose-test.i']),
        ((1,2,False), ['proteus-opt','--n-threads=2','-i','moose-test.i']),
        ((1,2,True), ['proteus-opt','--n-threads=2','-i','moose-test.i','--redirect-stdout']),
        ((2,2,False), ['mpirun','-np','2','proteus-opt','--n-threads=2','-i','moose-test.i']),
        ((2,2,True), ['mpirun','-np','2','proteus-opt','--n-threads=2','-i','moose-test.i','--redirect-stdout']),
    )
)
def test_assemble_arg_list_with_input(opts: tuple[int,int,bool],
                                     expected: str,
                                     runner: MooseRunner,
                                     input_path: Path) -> None:
    runner.set_run_opts(opts[0],opts[1],opts[2])
    assert runner.assemble_arg_list(input_path) == expected


def test_assemble_arg_list_err(runner: MooseRunner) -> None:
    with pytest.raises(RuntimeError) as err_info:
        runner.assemble_arg_list()

    msg, = err_info.value.args
    assert msg == 'No input file specified, set one using set_input_file or by passing on into this function.'


def test_assemble_arg_list_err_with_input(runner: MooseRunner,
                                         input_noexist: Path) -> None:
    with pytest.raises(FileNotFoundError) as err_info:
        runner.assemble_arg_list(input_noexist)

    msg, = err_info.value.args
    assert msg == 'Input file does not exist.'


@pytest.mark.parametrize(
    ('opts','stdout_exist'),
    (
        ((1,1,False), (False,False)),
        ((1,4,False), (False,False)),
        ((1,4,True), (True,False)),
        ((2,4,False), (False,False)),
        ((2,4,True), (True,True)),
    )
)
def test_run(opts: tuple[int,int,bool],
             stdout_exist: tuple[bool,bool],
             input_runner: MooseRunner) -> None:
    input_runner.set_run_opts(opts[0],opts[1],opts[2])
    input_runner.run()

    assert os.path.isfile(input_runner.get_output_path()) is True, 'No exodus output.' # type: ignore
    assert os.path.isfile(input_runner.get_input_dir() / 'stdout.processor.0') == stdout_exist[0], 'stdout.processor.0 does not exist.' # type: ignore
    assert os.path.isfile(input_runner.get_input_dir() / 'stdout.processor.1') == stdout_exist[1], 'stdout.processor.1 does not exist.' # type: ignore
    if opts[2]: # If there is a stdout it can be read to check for convergence
        check_path = input_runner.get_input_dir() / 'stdout.processor.0' # type: ignore
        assert hc.check_solve_converged(check_path) >= 1, 'Solve has not converged.'


def test_run_broken(runner: MooseRunner, input_broken: Path) -> None:
    runner.set_run_opts(1,4,True)
    runner.run(input_broken)

    stdout_file = runner.get_input_dir() / 'stdout.processor.0' # type: ignore

    assert os.path.isfile(runner.get_output_path()) is False # type: ignore
    assert os.path.isfile(stdout_file) is True # type: ignore

    assert hc.check_solve_error(stdout_file) >= 1, 'Error string not found in stdout'
    assert hc.check_solve_converged(stdout_file) == 0, 'Solve converged when it should have errored'


def test_run_noexist(runner: MooseRunner, input_noexist: Path) -> None:
    with pytest.raises(FileNotFoundError) as err_info:
        runner.run(input_noexist)

    msg, = err_info.value.args
    assert msg == 'Input file does not exist.'


def test_run_with_input(runner: MooseRunner, input_path: Path) -> None:
    runner.set_run_opts(1,4,True)
    runner.run(input_path)

    assert os.path.isfile(runner.get_output_path()) is True, 'Exodus output does not exist when solve should have run' # type: ignore
    assert os.path.isfile(runner.get_input_dir() / 'stdout.processor.0') is True, 'Stdout does not exist when it should.' # type: ignore
    assert hc.check_solve_converged(runner.get_input_dir() / 'stdout.processor.0') >= 1, 'Solve did not converge when it should have.' # type: ignore