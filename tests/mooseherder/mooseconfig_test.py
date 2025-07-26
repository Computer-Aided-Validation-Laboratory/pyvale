'''
==============================================================================
TEST: MooseConfig

Authors: Lloyd Fletcher
==============================================================================
'''
import os
from pathlib import Path
import pytest
from mooseherder.mooseconfig import MooseConfig, MooseConfigError
import tests.herdchecker as hc


@pytest.fixture
def config_dict() -> dict[str, Path | str]:
    return {'main_path': hc.MOOSE_PATH,
            'app_path': hc.MOOSE_APP_PATH,
            'app_name': hc.MOOSE_APP_NAME}


@pytest.fixture
def config_dict_str() -> dict[str,str]:
    return {'main_path': str(hc.MOOSE_PATH),
            'app_path': str(hc.MOOSE_APP_PATH),
            'app_name': str(hc.MOOSE_APP_NAME)}


@pytest.fixture
def config_dict_broken_path() -> dict[str, Path | str]:
    return {'main_path': Path().home() / 'no-exist',
            'app_path': Path().home() / 'no-exist',
            'app_name': 'broken-opt'}


@pytest.fixture
def config_dict_broken_key() -> dict[str, Path | str]:
    return {'main_path': hc.MOOSE_PATH,
            'app_name': hc.MOOSE_APP_NAME}


@pytest.fixture
def config(config_dict: dict[str, Path | str]) -> MooseConfig:
    return MooseConfig(config_dict)


def test_init_config_blank() -> None:
    moose_config = MooseConfig()
    assert moose_config is not None


def test_init_config(config: MooseConfig) -> None:
    assert config is not None


def test_init_config_key_err(config_dict_broken_key: dict[str, Path | str]
                             ) -> None:
    with pytest.raises(MooseConfigError) as err_info:
        moose_config = MooseConfig(config_dict_broken_key)

    (msg,) = err_info.value.args
    assert msg == "Config dictionary must contain all keys: ['main_path', 'app_path', 'app_name']"


def test_init_config_path_err(config_dict_broken_path: dict[str, Path | str]
                              ) -> None:
    with pytest.raises(MooseConfigError) as err_info:
        moose_config = MooseConfig(config_dict_broken_path)

    (msg,) = err_info.value.args
    assert msg == "Main path to MOOSE does not exist. Check path at key 'main_path'."


def test_get_config(config: MooseConfig,
                    config_dict: dict[str, Path | str]) -> None:
    check_config = config.get_config()
    assert check_config == config_dict


def test_blank_config_get_err() -> None:
    moose_config = MooseConfig()

    with pytest.raises(MooseConfigError) as err_info:
        check_config = moose_config.get_config()

    (msg,) = err_info.value.args
    assert msg == 'Config dictionary must be initialised, load config file first.'


def test_convert_path_to_str(config_dict: dict[str, Path | str],
                             config_dict_str: dict[str, str]) -> None:
    conv_config = MooseConfig().convert_path_to_str(config_dict)
    assert conv_config == config_dict_str


def test_convert_path_to_str_blank() -> None:
    conv_config = MooseConfig().convert_path_to_str(None)
    assert conv_config is None


def test_convert_str_to_path(config_dict: dict[str, Path | str],
                             config_dict_str: dict[str, str]) -> None:
    conv_config = MooseConfig().convert_str_to_path(config_dict_str)
    assert conv_config == config_dict


def test_convert_str_to_path_blank() -> None:
    conv_config = MooseConfig().convert_str_to_path(None)
    assert conv_config is None


def test_save_config(config: MooseConfig) -> None:
    save_path = Path('tests/config/moose-config.json')
    config.save_config(save_path)
    assert save_path.is_file()
    os.remove(save_path)


def test_save_config_parent_dir_err(config: MooseConfig) -> None:
    save_path = Path('tests/no-exist/moose-config.json')
    with pytest.raises(MooseConfigError) as err_info:
        config.save_config(save_path)

    (msg,) = err_info.value.args
    assert msg == 'Parent path to save config file does not exist.'


def test_read_config(config_dict: dict[str, Path | str]) -> None:
    read_path = Path().cwd() / 'moose-config.json'
    read_config = MooseConfig().read_config(read_path)
    assert read_config.get_config() == config_dict


def test_read_config_exist_err() -> None:
    read_path = Path('tests/no-exist/moose-config-no-exist.json')
    with pytest.raises(MooseConfigError) as err_info:
        read_config = MooseConfig().read_config(read_path)

    (msg,) = err_info.value.args
    assert msg == 'MOOSE config file does not exist at: tests/no-exist/moose-config-no-exist.json.'


def test_read_config_key_err() -> None:
    read_path = Path('tests/config/moose-config-break-key.json')

    with pytest.raises(MooseConfigError) as err_info:
        read_config = MooseConfig().read_config(read_path)

    (msg,) = err_info.value.args
    assert msg == "Config dictionary must contain all keys: ['main_path', 'app_path', 'app_name']"


def test_read_config_path_err() -> None:
    read_path = Path('tests/config/moose-config-break-path.json')

    with pytest.raises(MooseConfigError) as err_info:
        read_config = MooseConfig().read_config(read_path)

    (msg,) = err_info.value.args
    assert msg == "Main path to MOOSE does not exist. Check path at key 'main_path'."
