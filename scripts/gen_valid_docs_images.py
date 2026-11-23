#!/usr/bin/env python3
# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Run validation tutorials and regenerate the PNGs embedded in their docs."""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Local pyvale-data/Data directory")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "src/pyvale/examples/validation"
    images_dir = repo_root / "images"
    static_dir = repo_root / "docs/source/_static"
    environment = os.environ.copy()
    environment.update(
        MPLBACKEND="Agg",
        PYVISTA_OFF_SCREEN="true",
        QT_QPA_PLATFORM="offscreen",
    )
    if args.data_dir is not None:
        data_dir = args.data_dir.expanduser().resolve()
        if not data_dir.is_dir():
            parser.error(f"Data directory does not exist: {data_dir}")
        environment["PYVALE_DATA"] = str(data_dir)

    examples = sorted(examples_dir.glob("ex[0-9][0-9]*_valid_*.py"))
    if not examples:
        parser.error("No validation examples found in this checkout.")

    with TemporaryDirectory(prefix="pyvale-valid-docs-") as scratch:
        work_dir = Path(scratch)
        figures: list[Path] = []
        for example in examples:
            image_names = re.findall(
                r"^# \.\. image:: /_static/(val_ex\d{2}_fig[^/\s]+\.png)$",
                example.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
            if not image_names:
                parser.error(f"No documentation figures referenced by {example.name}.")
            print(f"Generating figures for {example.stem}", flush=True)
            subprocess.run(
                (sys.executable, str(example)), cwd=work_dir, env=environment, check=True
            )
            output_dir = work_dir / "pyvale-output" / (
                "valid_" + example.stem.replace("_valid_", "_", 1)
            )
            for name in image_names:
                figure = output_dir / name
                if not figure.is_file():
                    parser.error(f"{example.name} did not generate {name}.")
                figures.append(figure)

        images_dir.mkdir(parents=True, exist_ok=True)
        static_dir.mkdir(parents=True, exist_ok=True)
        for figure in figures:
            image = images_dir / figure.name
            link = static_dir / figure.name
            target = Path(os.path.relpath(image, static_dir))
            if link.is_symlink():
                if link.readlink() != target:
                    parser.error(f"Unexpected image symlink: {link}")
            elif link.exists():
                parser.error(f"Cannot create image symlink over existing file: {link}")
            else:
                link.symlink_to(target)
            shutil.copyfile(figure, image)

    print(f"Generated {len(figures)} documentation figures in {images_dir}.")


if __name__ == "__main__":
    main()
