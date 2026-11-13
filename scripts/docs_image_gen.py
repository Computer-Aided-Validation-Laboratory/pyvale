#!/usr/bin/env python3
"""Generate standalone physics-simulation images for the documentation.

Run with ``.venv/bin/python scripts/docs_image_gen.py`` from the repository root.
The default Exodus dataset is resolved through PyVale's data API, which can
download and cache it. No solver is launched and no sensor overlays are added.
"""

import argparse
import json
from pathlib import Path

import pyvista as pv

import pyvale.data as dataset


def create_simulation_images(
    exodus_path: Path,
    output_dir: Path,
    *,
    field: str = "temperature",
    time_index: int = -1,
) -> None:
    """Render a scalar field and a mesh view, preserving native coordinates.

    Displacement warping is explicitly disabled. Only element blocks are shown;
    Exodus boundary/node sets are excluded. Field values and times retain their
    source units. A manifest records the selected dataset, field and time.
    """
    reader = pv.ExodusIIReader(exodus_path)
    reader.disable_displacements()
    reader.disable_all_point_arrays()
    if field not in reader.point_array_names:
        raise ValueError(
            f"Unknown nodal field {field!r}; choose from {reader.point_array_names}"
        )
    reader.enable_point_array(field)
    time_count = len(reader.time_values)
    if not -time_count <= time_index < time_count:
        raise ValueError(f"time_index must be between {-time_count} and {time_count - 1}")
    selected_index = time_index % time_count
    reader.set_active_time_point(selected_index)
    mesh = reader.read()["Element Blocks"].combine()
    if mesh.point_data[field].ndim != 1:
        raise ValueError(f"Expected a scalar nodal field, got {field!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("physics_simulation_field.png", "physics_simulation_mesh.png"):
        plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
        try:
            plotter.set_background("white")
            if filename == "physics_simulation_field.png":
                plotter.add_mesh(
                    mesh,
                    scalars=field,
                    cmap="viridis",
                    show_edges=True,
                    edge_color="#383838",
                    line_width=0.5,
                    lighting=False,
                    scalar_bar_args={
                        "title": f"{field} (native units)",
                        "color": "black",
                        "title_font_size": 22,
                        "label_font_size": 18,
                        "position_x": 0.2,
                        "width": 0.6,
                        "position_y": 0.06,
                    },
                )
            else:
                plotter.add_mesh(
                    mesh,
                    color="#d8e7ce",
                    show_edges=True,
                    edge_color="#383838",
                    lighting=False,
                )
            if mesh.bounds.z_min == mesh.bounds.z_max:
                plotter.view_xy()
            else:
                plotter.view_isometric()
            plotter.enable_parallel_projection()
            plotter.reset_camera()
            plotter.camera.zoom(1.15)
            plotter.screenshot(output_dir / filename)
        finally:
            plotter.close()

    manifest = {
        "exodus_path": str(exodus_path.resolve()),
        "field": field,
        "time_index": selected_index,
        "time_native_units": reader.time_values[selected_index],
        "displacement_warping": False,
        "coordinate_scaling": 1.0,
        "point_count": mesh.n_points,
        "cell_count": mesh.n_cells,
        "images": ["physics_simulation_field.png", "physics_simulation_mesh.png"],
    }
    (output_dir / "physics_simulation.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Select the dataset, scalar field, time sample and output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Defaults to PyVale's case18 Exodus")
    parser.add_argument("--field", default="temperature")
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "pyvale-output/docs-images",
    )
    args = parser.parse_args()
    exodus_path = (
        args.input if args.input is not None else dataset.thermomechanical_2d_path()
    )
    create_simulation_images(
        exodus_path,
        args.output_dir,
        field=args.field,
        time_index=args.time_index,
    )
    print(f"Created physics-simulation images in {args.output_dir}")


if __name__ == "__main__":
    main()
