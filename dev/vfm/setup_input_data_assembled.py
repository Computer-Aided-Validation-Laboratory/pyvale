"""Prepare any published ``fe-data/raw`` bundle using the generic PyVale adapter."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pyvale.vfm import (
    AssembledDataConfig,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    process_input_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_data", type=Path, help="Published fe-data/raw directory")
    parser.add_argument("prepared", type=Path, help="Destination prepared directory")
    parser.add_argument("--thickness-mm", type=float, required=True)
    parser.add_argument("--loading", choices=("tensile-x", "tensile-y", "free"), default="tensile-x")
    args = parser.parse_args()
    free = Edge(EEdgeCondition.Free, EEdgeCondition.Free)
    if args.loading == "tensile-x":
        edges = EdgeConditions(
            min_x_edge=Edge(EEdgeCondition.Fixed, EEdgeCondition.Free),
            max_x_edge=Edge(EEdgeCondition.Traction, EEdgeCondition.Fixed),
            min_y_edge=free,
            max_y_edge=free,
        )
    elif args.loading == "tensile-y":
        edges = EdgeConditions(
            min_x_edge=free,
            max_x_edge=free,
            min_y_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Fixed),
            max_y_edge=Edge(EEdgeCondition.Fixed, EEdgeCondition.Traction),
        )
    else:
        edges = EdgeConditions(free, free, free, free)
    config = AssembledDataConfig(
        data_dir=args.raw_data,
        thickness=args.thickness_mm,
        edge_conditions=edges,
    )
    output = process_input_data(config, output_root=args.prepared, timestamped=False)
    maps = args.raw_data / "known_parameter_maps.npz"
    if maps.is_file():
        shutil.copy2(maps, args.prepared / maps.name)
    metadata = args.raw_data / "metadata.yaml"
    if metadata.is_file():
        shutil.copy2(metadata, args.prepared / "raw_metadata.yaml")
    (args.prepared / "README.md").write_text(
        "# Prepared PyVale data\n\n"
        "Generated from the solver-independent `fe-data/raw` bundle using "
        "PyVale's generic assembled-data adapter. See `raw_metadata.yaml` "
        "and the parent dataset provenance for source details.\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
