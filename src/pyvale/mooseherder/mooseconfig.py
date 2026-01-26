
# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
from typing import Self
import json
from pathlib import Path


class MooseConfig:
    """Moose configuration class that handles 1) the path to the main moose
    build, 2) the path to the moose app, and 3) the name of the app to be used
    to construct the command string. These are stored as a dictionary keyed
    with 'main_path', 'app_path' and 'app_name'. This class can also write and
    read json files containing the moose config.
    """
    __slots__ = ("_required_keys","_config")

    def __init__(self, config: dict[str,Path | str] | None = None) -> None:

        self._required_keys = ['main_path','app_path','app_name']

        if config is not None:
            self._check_config_valid(config)

        self._config = config


    def get_config(self) -> dict[str,Path | str]:
        """get_config: returns the config dictionary after checking it is
        valid.

        Returns
        -------
        dict[str, Path | str]
            dictionary containing the moose config.

        """
        self._check_config_valid(self._config)
        return self._config # type: ignore


    def _check_config_valid(self,
                            config: dict[str,Path | str] | None = None
                            ) -> None:
        """_check_config_valid: helper function to check if the moose config
        is valid.

        Parameters
        ----------
        config : dict[str, Path | str] | None = None
            Dictionary containing the moose config. Defaults to None.

        Raises
        ------
        MooseConfigError
            Dicitionary not initialised.
        MooseConfigError
            Dicitionary does not contain the required keys.
        MooseConfigError
            Path to MOOSE does not exist.
        MooseConfigError
            Path to MOOSE app does not exist.

        """
        if config is None:
            raise MooseConfigError(
                'Config dictionary must be initialised, load config file first.')

        for kk in self._required_keys:
            if kk not in config:
                raise MooseConfigError(
                    f'Config dictionary must contain all keys: {self._required_keys}')

        if not config['main_path'].is_dir(): # type: ignore
            raise MooseConfigError(
                "Main path to MOOSE does not exist. Check path at key 'main_path'.")

        if not config['app_path'].is_dir(): # type: ignore
            raise MooseConfigError(
                "MOOSE app path does not exist. Check path at key: 'app_path'.")


    def convert_path_to_str(self, in_config: dict[str,Path | str] | None
                             ) -> dict[str,str] | None:
        """convert_path_to_str: converts all paths in the config dictionary to
        strings so that it can be saved to json.

        Parameters
        ----------
        in_config : dict[str, Path | str] | None
            Dictionary containing the moose config. Defaults to None.

        Returns
        -------
        dict[str,str] | None
            as input with Paths converted to strings.

        """
        if in_config is None:
            return None

        conv_config = dict({})
        for kk in in_config:
            conv_config[kk] = str(in_config[kk])

        return conv_config


    def convert_str_to_path(self, in_config: dict[str,str] | None = None
                             ) -> dict[str, Path | str] | None:
        """convert_str_to_path: helper function to convert string to Path for
        readin in json dictionary. Does not check if the paths exist.

        Parameters
        ----------
        in_config : dict[str, Path | str] | None
            Dictionary containing the moose config. Defaults to None.

        Returns
        -------
        dict[str, Path | str] | None
            as in_config but strings to main_path
            and app_path are converted to Path.

        """
        if in_config is None:
            return None

        conv_config = dict({})
        for kk in in_config:
            if 'path' in kk:
                conv_config[kk] = Path(in_config[kk])
            else:
                conv_config[kk] = in_config[kk]

        return conv_config


    def save_config(self, config_path: Path) -> None:
        """save_config: saves the moose config dictionary as a json file in the
        specified path.

        Parameters
        ----------
        config_path : Path
            path and file name with extension .json to save the moose config
            dictionary.

        Raises
        ------
        MooseConfigError
            Parent path to save config file does not exist.

        """
        if not config_path.parent.is_dir():
            raise MooseConfigError('Parent path to save config file does not exist.')

        with open(config_path, 'w', encoding='utf-8') as cf:
            config_strs = self.convert_path_to_str(self._config)
            json.dump(config_strs, cf, indent=4)


    def read_config(self, config_path: Path) -> Self:
        """read_config: reads the moose json configuration file at the
        specified path. Checks if the configuration is valid and raises a
        MooseConfigError if it is not.

        Parameters
        ----------
        config_path : Path
            path to the json config file containing the

        Returns
        -------
        Self
            returns a MooseConfig object allowing the config to be
            initialised by directly loading a json config file.

        Raises
        ------
        MooseConfigError
            MOOSE config file does not exist.

        """
        if not config_path.is_file():
            raise MooseConfigError(
                f'MOOSE config file does not exist at: {str(config_path)}.')

        with open(config_path, 'r', encoding='utf-8') as cf:
            config_strs = json.load(cf)
            config_paths = self.convert_str_to_path(config_strs)

        self._check_config_valid(config_paths)

        self._config = config_paths
        return self


class MooseConfigError(Exception):
    """MooseConfigError: custom error class for flagging errors with the moose
    configuration.

    Parameters
    ----------

    Returns
    -------

    """
