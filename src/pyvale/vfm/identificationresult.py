import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml

from pyvale.vfm.spatialparam import ISpatialParameterisation


@dataclass(slots=True)
class ParameterisationSnapshot:
    """
    Snapshot of a single spatial parameterisation and its degree-of-freedom
    values, captured at the end of an identification phase.
    """

    parameterisation: ISpatialParameterisation
    """Copy of the spatial parameterisation as it was at the end of the phase"""

    dof_values: list[float]
    """
    The value of each degree of freedom of ``parameterisation``, in the order
    returned by its ``collect_degrees_of_freedom``
    """


@dataclass(slots=True)
class PhaseSnapshot:
    """
    Snapshot of one identification phase, captured at the end of the phase.

    Stores, for each constitutive parameter, the spatial parameterisations that
    represent it together with their degree-of-freedom values
    """

    spatial_parameterisations: dict[str, list[ParameterisationSnapshot]]
    """
    Mapping from constitutive parameter name to the snapshots of its spatial
    parameterisations, in definition order
    """

    @classmethod
    def from_spatial_parameterisations(
        cls,
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    ) -> "PhaseSnapshot":
        """
        Build a snapshot from a phase's current spatial parameterisations.

        Each parameterisation is deep-copied so that later phases cannot mutate
        the captured state, and its degree-of-freedom values are recorded.
        """
        snapshot: dict[str, list[ParameterisationSnapshot]] = {}
        for name, parameterisation_list in spatial_parameterisations.items():
            snapshot[name] = [
                ParameterisationSnapshot(
                    parameterisation=copy.deepcopy(parameterisation),
                    dof_values=[
                        dof.value
                        for dof in parameterisation.collect_degrees_of_freedom()
                    ],
                )
                for parameterisation in parameterisation_list
            ]
        return cls(snapshot)


@dataclass(slots=True)
class IdentificationHistory:
    """
    Ordered, per-phase history of an identification run.

    Holds one :class:`PhaseSnapshot` per identification phase, in the order the
    phases were executed. Each snapshot is taken at the end of its phase
    """

    phases: list[PhaseSnapshot] = field(default_factory=list)
    """Per-phase snapshots, in execution order"""


@dataclass(slots=True)
class IdentificationResult:
    """
    Result of a VFM identification run.

    Combines the final identified parameter maps with the per-phase history of
    the run
    """

    parameter_maps: dict[str, npt.NDArray[np.float64]]
    """Final identified parameter map for each constitutive parameter name"""

    history: IdentificationHistory
    """Per-phase history of the identification"""

    def save_to_yaml(self, output_dir: str | Path | None = None) -> Path:
        """
        Save this result into ``output_dir``.

        Writes an ``identification_result.yaml`` describing the result, with
        each final parameter map saved as a sibling ``.npy`` file referenced
        from the yaml. The history is written inline: for each phase, and for
        each constitutive parameter, every spatial parameterisation is recorded
        as its type name plus its list of degree-of-freedom values. The
        directory is created if it does not exist.

        Parameters
        ----------
        output_dir : str | Path | None, optional
            Directory to write the result into. When ``None`` (the default) a
            new ``vfm-identification-result_{timestamp}`` directory is created
            in the current directory

        Returns
        -------
        Path
            Path of the written ``identification_result.yaml`` file
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            output_dir = f"vfm-identification-result_{timestamp}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        parameter_map_files: dict[str, str] = {}
        for name, parameter_map in self.parameter_maps.items():
            filename = f"parameter_map_{name}.npy"
            np.save(
                output_dir / filename,
                np.asarray(parameter_map, dtype=np.float64),
            )
            parameter_map_files[name] = filename

        phases = []
        for phase_snapshot in self.history.phases:
            spatial_parameterisations = {
                name: [
                    {
                        "parameterisation": type(
                            snapshot.parameterisation
                        ).__name__,
                        "dof_values": [
                            float(value) for value in snapshot.dof_values
                        ],
                    }
                    for snapshot in snapshots
                ]
                for name, snapshots in
                phase_snapshot.spatial_parameterisations.items()
            }
            phases.append(
                {"spatial_parameterisations": spatial_parameterisations}
            )

        content = {
            "parameter_maps": parameter_map_files,
            "history": {"phases": phases},
        }

        result_file = output_dir / "identification_result.yaml"
        result_file.write_text(
            yaml.safe_dump(content, sort_keys=False),
            encoding="utf-8",
        )

        return result_file
