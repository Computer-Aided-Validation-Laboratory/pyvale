'''
==============================================================================
TEST: DirectoryManager

Authors: Lloyd Fletcher
==============================================================================
'''

from pathlib import Path
import pytest
from mooseherder.directorymanager import DirectoryManager
import tests.herdchecker as hc


@pytest.fixture
def dir_manager() -> DirectoryManager:
    return DirectoryManager(hc.NUM_DIRS)


@pytest.fixture(autouse=True)
def setup_teardown(dir_manager):
    yield
    dir_manager.clear_dirs()


def test_init_dir_manager(dir_manager: DirectoryManager) -> None:
    assert dir_manager is not None
    assert dir_manager._n_dirs == hc.NUM_DIRS
    assert dir_manager._base_dir == Path().cwd()
    assert dir_manager._sub_dir == 'sim-workdir'


def test_set_sub_dir_name(dir_manager: DirectoryManager) -> None:
    new_name = 'new-workdir'
    dir_manager.set_sub_dir_name(new_name)
    assert dir_manager._sub_dir == new_name


def test_set_base_dir(dir_manager: DirectoryManager) -> None:
    dir_manager.set_base_dir(hc.BASE_DIR)
    assert dir_manager._base_dir == hc.BASE_DIR


def test_set_base_dir_err(dir_manager: DirectoryManager) -> None:
    base_dir = Path('no_exist/')
    with pytest.raises(FileExistsError) as errinfo:
        dir_manager.set_base_dir(base_dir)
    msg, = errinfo.value.args
    assert msg == "Specified base directory does not exist."


def test_create_dirs(dir_manager: DirectoryManager) -> None:
    dir_manager.create_dirs()
    for dd in range(hc.NUM_DIRS):
        run_dir = dir_manager._base_dir / (dir_manager._sub_dir+'-'+str(dd+1))
        assert run_dir.is_dir()


def test_clear_dirs(dir_manager: DirectoryManager) -> None:
    dir_manager.create_dirs()
    dir_manager.clear_dirs()
    for dd in range(hc.NUM_DIRS):
        run_dir = dir_manager._base_dir / (dir_manager._sub_dir+'-'+str(dd+1))
        assert run_dir.is_dir() == False


@pytest.mark.parametrize(
    ('dir_num','expected'),
    (
        (-1,0),
        (0,0),
        (1,1),
        (4,0),
    )
)
def test_get_run_dir(dir_num: int,
                     expected: int,
                     dir_manager: DirectoryManager) -> None:
    check_path = dir_manager.get_run_dir(dir_num)
    assert check_path == dir_manager._run_dirs[expected]