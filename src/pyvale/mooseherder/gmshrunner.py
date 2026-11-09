# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Generate meshes from geometry files or user Python scripts in subprocesses."""

import json
import math
import os
import sys
from pathlib import Path

from .simrunner import (
    InputFiles,
    InputValues,
    RunnerBackend,
    RunResult,
    SimRunner,
    _check_backend_exe,
    _check_output_file_name,
    conv_args_dict_to_list,
    execute,
    stage_input_files,
)


class GmshRunner(SimRunner):
    """Generate a mesh from a .geo file or a user-supplied Python script.

    Notes
    -----
    Default .geo execution uses Python Gmsh in an isolated worker process. Scripts
    receive --config followed by a JSON path containing input_params and output_path,
    and must write the requested mesh. Native geometry parameters follow Gmsh's
    -setnumber/-setstring convention; use DefineConstant for overridable values.
    Existing Mesh/Save commands are supported, but Exit commands must be removed.
    """

    def __init__(
        self,
        *,
        geo_file: Path | None = None,
        py_script_file: Path | None = None,
        gmsh_exe_file: Path | None = None,
        mesh_dims: int = 2,
        mesh_order: int | None = None,
        timeout: float = 3600.0,
        output_file_name: str = "mesh.msh",
        msh_version: float = 2.2,
    ) -> None:
        """Configure a geometry source, mesh settings and optional custom executable.

        Parameters
        ----------
        geo_file : Path | None, optional
            Default .geo input. Mutually exclusive with py_script_file. Defaults to None.
        py_script_file : Path | None, optional
            Default Python mesh script implementing the JSON CLI contract. Defaults
            to None. A run may instead supply an explicit input path.
        gmsh_exe_file : Path | None, optional
            Custom Gmsh executable for .geo inputs only. None selects Python Gmsh.
        mesh_dims : int, optional
            Dimension to generate when a geometry script has not already meshed its
            model: 1, 2 or 3. Defaults to 2. Passed to user scripts in their config.
        mesh_order : int | None, optional
            Positive mesh order. None preserves a .geo script's existing order.
            User scripts receive this setting and own their meshing options.
        timeout : float, optional
            Maximum subprocess runtime in seconds. Defaults to 3600 (1 hour).

        output_file_name : str, optional
            Output basename, including the export format extension. Defaults to
            "mesh.msh". Gmsh chooses the export format from this extension.
        msh_version : float, optional
            MSH format version, default 2.2 for MOOSE compatibility. Applies only
            to .msh output; other extensions select their own export format.

        Raises
        ------
        ValueError
            Sources conflict, mesh dimensions/order, timeout or output settings
            are invalid.
        FileNotFoundError
            The custom executable is missing.
        PermissionError
            The custom executable cannot be executed.
        RuntimeError
            Python Gmsh or its shared library cannot be imported.
        """
        if geo_file is not None and py_script_file is not None:
            raise ValueError("Specify geo_file or py_script_file, not both.")

        if type(mesh_dims) is not int or mesh_dims not in (1, 2, 3):
            raise ValueError("mesh_dims must be 1, 2 or 3.")

        if mesh_order is not None and (type(mesh_order) is not int or mesh_order < 1):
            raise ValueError("mesh_order must be a positive integer.")

        if timeout <= 0:
            raise ValueError("timeout must be positive.")

        self.input_file = geo_file if geo_file is not None else py_script_file
        self.gmsh_exe_file = (
            gmsh_exe_file.resolve() if gmsh_exe_file is not None else None
        )
        self.mesh_dims = mesh_dims
        self.mesh_order = mesh_order
        _check_output_file_name(output_file_name)
        if output_file_name in {
            "run.log",
            "command.json",
            "input_params.json",
            "mesh_config.json",
        }:
            raise ValueError("output_file_name conflicts with runner metadata.")
        if not Path(output_file_name).suffix:
            raise ValueError("output_file_name must have a mesh format extension.")
        if (
            isinstance(msh_version, bool)
            or not isinstance(msh_version, (int, float))
            or not math.isfinite(msh_version)
            or msh_version <= 0
        ):
            raise ValueError("msh_version must be a positive finite number.")
        self.timeout = timeout
        self.output_file_name = output_file_name
        self.msh_version = msh_version
        self.get_backend()

    def get_backend(self) -> RunnerBackend:
        """Check the Python Gmsh package or explicit Gmsh executable.

        Returns
        -------
        RunnerBackend
            Executable path for custom Gmsh, or the Python gmsh.py module path.
            Checks run at construction and before every mesh operation. No GUI
            is opened and no Gmsh model is initialised by this check.

        Raises
        ------
        FileNotFoundError
            The custom executable is missing.
        PermissionError
            The custom executable lacks OS execution permission.
        RuntimeError
            Python Gmsh or its shared library cannot be imported.
        """
        if self.gmsh_exe_file is not None:
            return RunnerBackend(_check_backend_exe(self.gmsh_exe_file))
        try:
            import gmsh
        except (ImportError, OSError) as err:
            raise RuntimeError(f"Python Gmsh backend is unavailable: {err}") from err
        return RunnerBackend(Path(gmsh.__file__).resolve())

    def run(
        self,
        input_files: InputFiles | None = None,
        *,
        output_dir: Path,
        input_params: dict[str, InputValues] | None = None,
        input_root: Path | None = None,
    ) -> RunResult:
        """Stage a geometry/script input and generate the requested mesh.

        Parameters
        ----------
        input_files : Path | list[Path] | tuple[Path, ...] | None, optional
            One .geo or .py path. None uses the geometry/script set at construction;
            lists and tuples are not accepted by this runner.
        output_dir : Path
            New working directory outside the input tree.
        input_params : dict[str, InputValues] | None, optional
            Gmsh geometry scalars or user-script values, in the geometry's units.
            Geometry sequences are rejected; scripts receive values through JSON.
        input_root : Path | None, optional
            Geometry/script tree to snapshot, including its relative assets.
            None uses the selected input file's parent directory.

        Returns
        -------
        RunResult
            Successful primary output, working directory, invocation and log paths.

        Raises
        ------
        ValueError
            Inputs or run configuration are invalid or ambiguous.
        FileNotFoundError
            An input file is missing.
        FileExistsError
            The output directory already exists.
        RuntimeError
            A backend is unavailable, a process fails/times out, or expected output
            is absent or empty.
        TypeError
            A .geo parameter is not scalar or a parameter value is unsupported.

        Notes
        -----
        The primary output uses output_file_name; its extension selects the
        format. msh_version controls .msh output, defaulting to MSH 2.2 for MOOSE
        compatibility. Supplied scripts must write the configured output_path; custom
        executables cannot execute Python mesh scripts. Missing custom executables
        never fall back to the packaged backend.
        """
        source = input_files if input_files is not None else self.input_file

        if not isinstance(source, Path) or source.suffix not in (".geo", ".py"):
            raise ValueError("Specify one .geo geo_file or .py mesh py_script_file.")

        if source.suffix == ".py" and self.gmsh_exe_file is not None:
            raise ValueError("A custom Gmsh executable cannot execute Python scripts.")

        input_params = dict(input_params or {})
        conv_args_dict_to_list(input_params)

        if source.suffix == ".geo" and any(
            isinstance(value, (list, tuple)) for value in input_params.values()
        ):
            raise TypeError("Gmsh geometry input parameters must be scalar.")

        backend = self.get_backend()

        directory = output_dir.resolve()
        staged = stage_input_files(source, directory, input_root)[0]
        output_file = directory / self.output_file_name
        config = directory / "mesh_config.json"
        config.write_text(
            json.dumps(
                {
                    "input_path": str(staged),
                    "output_path": str(output_file),
                    "input_params": input_params,
                    "mesh_dims": self.mesh_dims,
                    "mesh_order": self.mesh_order,
                    "msh_version": self.msh_version,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if staged.suffix == ".py":
            command = [sys.executable, str(staged), "--config", str(config)]

        elif self.gmsh_exe_file is not None:
            command = [
                str(backend.backend_file),
                str(staged),
                f"-{self.mesh_dims}",
                "-o",
                str(output_file),
                "-format",
                "auto",
                "-setnumber",
                "Mesh.MshFileVersion",
                str(self.msh_version),
            ]

            if self.mesh_order is not None:
                command.extend(("-order", str(self.mesh_order)))

            for name, value in input_params.items():
                if isinstance(value, (int, float)):
                    command.extend(
                        (
                            "-setnumber",
                            name,
                            str(int(value) if isinstance(value, bool) else value),
                        )
                    )
                elif isinstance(value, str):
                    command.extend(("-setstring", name, value))
                else:
                    raise TypeError("Gmsh geometry input parameters must be scalar.")
        else:
            # An absolute worker path works for source trees and installed wheels,
            # and does not import the whole PyVale package in the child process.
            worker = Path(__file__).with_name("_gmshworker.py")
            command = [sys.executable, str(worker), str(config)]

        log = execute(
            command,
            directory,
            input_params,
            env_vars=os.environ.copy(),
            timeout=self.timeout,
        )

        if not output_file.is_file() or output_file.stat().st_size == 0:
            raise RuntimeError(f"Gmsh did not produce {output_file}; see {log}")

        return RunResult(output_file, directory, tuple(command), log)
