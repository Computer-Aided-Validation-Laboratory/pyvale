# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import subprocess
from pathlib import Path
from pyvale.mooseherder.simrunner import SimRunner


class GmshRunner(SimRunner):
    """Used to call gmsh to create a mesh file to be used to run a finite
    element simulation. Implements the SimRunner abstract interface so that it
    can be used by the herd workflow manager.
    """

    __slots__ = ("_gmsh_app","_input_path","_arg_list")

    def __init__(self, gmsh_path: Path | None = None):
        """
        Parameters
        ----------
        gmsh_path : Path | None, optional
            Path to the gmsh executable, by default None
        """

        if gmsh_path is None:
            self._gmsh_app = None
        else:
            self.set_gmsh_app(gmsh_path)

        self._input_path = None
        self._arg_list = []

    def set_gmsh_app(self, gmsh_app: Path) -> None: # type: ignore
        """Sets path to the gmsh app.

        Parameters
        ----------
        gmsh_app : Path
            full path to the gmsh app.

        Raises
        ------
        FileNotFoundError
            gmsh app does not exist at the specified path.

        """
        if not gmsh_app.exists():
            raise FileNotFoundError('Gmsh app not found at given path.')

        self._gmsh_app = gmsh_app


    def get_input_file(self) -> Path | None:
        """get_input_path: the path to the input file to run gmsh with.

        Returns
        -------
        Path | None
            path to the gmsh *.geo file.

        """
        return self._input_path


    def set_input_file(self, input_path: Path) -> None:
        """Sets the input geo file for gmsh.

        Parameters
        ----------
        input_file : Path
            Full path to the gmsh *.geo input file.

        Returns
        -------

        Raises
        ------
        FileNotFoundError
            Not a .geo file
        FileNotFoundError
            Geo file does not exist

        """
        if input_path.suffix != '.geo':
            raise FileNotFoundError('Incorrect file type. Must be *.geo.')

        if not input_path.exists():
            raise FileNotFoundError('Specified gmsh geo file does not exist.')

        self._input_path = input_path


    def run(self, input_file: Path | None = None, parse_only: bool = True) -> None:
        """Run the geo file to create the mesh.

        Parameters
        ----------
        input_file : Path | None
            Path to the .geo file containing the input.
            Can also be preset using set_input_file. Defaults to "" and ises
            the input file specified using set_input_file.
        parse_only: bool :
             (Default value = True)

        Returns
        -------

        Raises
        ------
        RuntimeError
            the path to the gmsh app is empty and must be
            specified first.
        RuntimeError
            the input file string is empty and must be specified
            first.

        """
        if input_file is not None:
            self.set_input_file(input_file)

        if self._gmsh_app is None:
            raise RuntimeError("Specify the full path to the gmsh app before calling run.")

        if self._input_path is None:
            raise RuntimeError("Specify input *.geo file before running gmsh.")

        arg_list = [str(self._gmsh_app)]
        if parse_only is True:
            arg_list = arg_list+["-parse_and_exit"]

        self._arg_list = arg_list + [str(self._input_path)]

        print(f'arg_list={self._arg_list}')

        subprocess.run(self._arg_list,
                       shell=False,
                       check=False)


    def get_output_path(self) -> Path | None:
        """get_output_path: default return None for gmsh as there is no output
        to be read after the simulation has run. This information is stored in
        the exodus.

        Parameters
        ----------

        Returns
        -------
        Path | None
            Default returns None.

        """
        return None

