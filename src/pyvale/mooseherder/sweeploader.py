# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Load completed sweep manifests without rerunning simulations."""

import json
from pathlib import Path

from pyvale.dataio.simdata import SimData, SimLoadConfig

from .exodusloader import ExodusLoader
from .simrunner import InputValues


class SweepLoader:
    """Read completed archive or reusable-evaluation manifests without rerunning them.

    Attributes
    ----------
    manifest : Path
        Absolute sweep.json or evaluation.json path.
    steps : list[str]
        Step names in execution order.
    case_input_params : list[dict[str, dict[str, InputValues]]]
        Submitted parameter dictionaries in case order.
    outputs : list[list[Path]]
        Resolved output paths, indexed first by case and then by step.

    Notes
    -----
    Relative paths in a manifest remain valid after relocating the whole sweep.
    A reusable evaluation manifest describes only that ID's current evaluation;
    read its outputs before a new evaluation replaces the directory.
    """

    def __init__(self, manifest: Path) -> None:
        """Load a completed sweep or single-evaluation manifest.

        Parameters
        ----------
        manifest : Path
            Existing JSON manifest produced after successful execution.

        Raises
        ------
        FileNotFoundError
            The manifest does not exist, including for an incomplete evaluation.
        ValueError
            The file does not contain valid JSON.
        KeyError
            Required steps, case_input_params or outputs metadata is absent.
        """
        self.manifest = manifest.resolve()

        data = json.loads(self.manifest.read_text(encoding="utf-8"))

        self.steps: list[str] = data["steps"]
        self.case_input_params: list[dict[str, dict[str, InputValues]]] = data[
            "case_input_params"
        ]

        self.outputs: list[list[Path]] = [
            [self.manifest.parent / p for p in row] for row in data["outputs"]
        ]

    def load_sim_data(
        self,
        step: str,
        config: SimLoadConfig | None = None,
    ) -> list[SimData]:
        """Load a named step's Exodus files into independent in-memory containers.

        Parameters
        ----------
        step : str
            Name of an Exodus-producing step in the manifest.
        config : SimLoadConfig | None, optional
            Fields, geometry and time indices to load. None loads all available data.

        Returns
        -------
        list[SimData]
            Selected simulation data in original case order. Each underlying file is
            closed before this method returns; arrays remain valid afterward.

        Raises
        ------
        ValueError
            The named step is absent from the manifest.
        OSError
            A referenced Exodus output cannot be opened or read.
        """
        index = self.steps.index(step)
        results = []

        for row in self.outputs:
            with ExodusLoader(row[index]) as reader:
                if config is None:
                    results.append(reader.load_all_sim_data())
                else:
                    results.append(reader.load_sim_data(config))

        return results
