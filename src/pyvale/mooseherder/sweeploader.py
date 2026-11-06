"""Load completed sweep manifests without rerunning simulations."""

import json
from pathlib import Path

from pyvale.dataio.simdata import SimData, SimLoadConfig

from .exodusloader import ExodusLoader
from .simrunner import MooseValue


class SweepLoader:
    """Read a sweep.json manifest; output paths remain valid after moving a sweep."""

    def __init__(self, manifest: Path) -> None:
        self.manifest = manifest.resolve()
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.steps: list[str] = data["steps"]
        self.cases: list[dict[str, dict[str, MooseValue]]] = data["cases"]
        self.outputs: list[list[Path]] = [
            [self.manifest.parent / p for p in row] for row in data["outputs"]
        ]

    def load_sim_data(
        self,
        step: str,
        config: SimLoadConfig | None = None,
    ) -> list[SimData]:
        """Read Exodus outputs from one named step, in original case order."""
        index = self.steps.index(step)
        results = []
        for row in self.outputs:
            reader = ExodusLoader(row[index])
            try:
                if config is None:
                    results.append(reader.load_all_sim_data())
                else:
                    results.append(reader.load_sim_data(config))
            finally:
                reader.close()
        return results
