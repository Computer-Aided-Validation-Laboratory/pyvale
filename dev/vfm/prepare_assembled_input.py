"""Prepare a solver-independent assembled-array bundle for VFM.

The raw directory must contain ``x.npy``, ``y.npy``, ``strain.npy``,
``force.npy`` and ``time.npy``. An optional ``specimen_mask.npy`` controls the
region of interest; otherwise finite strain values define it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pyvale.vfm import (
    AssembledDataConfig,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    process_input_data,
)


def main() -> None:
    args = _parse_args()
    config = AssembledDataConfig(
        data_dir=args.raw_data,
        thickness=args.thickness_mm,
        edge_conditions=_edge_conditions(args.loading),
    )
    experiment_data_file = process_input_data(
        config,
        output_root=args.prepared,
        timestamped=False,
    )

    _copy_optional_provenance(args.raw_data, args.prepared)

    experiment_data = ExperimentData.load_from_file(experiment_data_file)

    specimen_mask = (
        experiment_data.specimen_geometry.region_of_interest.sample_specimen_mask(
            experiment_data.specimen_geometry.x,
            experiment_data.specimen_geometry.y,
        )
    )
    print(f"Prepared: {experiment_data_file}")
    print(f"Strain shape: {experiment_data.strain.shape}")
    print(f"ROI points: {int(specimen_mask.sum())}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_data", type=Path, help="Directory of assembled .npy arrays")
    parser.add_argument("prepared", type=Path, help="Destination prepared directory")
    parser.add_argument("--thickness-mm", type=float, required=True)
    parser.add_argument(
        "--loading",
        choices=("tensile-x", "tensile-y", "free"),
        default="tensile-x",
        help="Boundary-condition convention for the prepared experiment",
    )
    return parser.parse_args()


def _edge_conditions(loading: str) -> EdgeConditions:
    free = Edge(EEdgeCondition.Free, EEdgeCondition.Free)
    if loading == "tensile-x":
        return EdgeConditions(
            min_x_edge=Edge(EEdgeCondition.Fixed, EEdgeCondition.Free),
            max_x_edge=Edge(EEdgeCondition.Traction, EEdgeCondition.Fixed),
            min_y_edge=free,
            max_y_edge=free,
        )
    if loading == "tensile-y":
        return EdgeConditions(
            min_x_edge=free,
            max_x_edge=free,
            min_y_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Fixed),
            max_y_edge=Edge(EEdgeCondition.Fixed, EEdgeCondition.Traction),
        )
    return EdgeConditions(free, free, free, free)


def _copy_optional_provenance(raw_data: Path, prepared: Path) -> None:
    optional_files = {
        "known_parameter_maps.npz": "known_parameter_maps.npz",
        "metadata.yaml": "raw_metadata.yaml",
    }
    for source_name, destination_name in optional_files.items():
        source = raw_data / source_name
        if source.is_file():
            shutil.copy2(source, prepared / destination_name)

    (prepared / "README.md").write_text(
        "# Prepared PyVale data\n\n"
        "Generated from a solver-independent assembled-array bundle using "
        "PyVale's generic assembled-data adapter. See `raw_metadata.yaml`, "
        "when present, and the source dataset for provenance.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()