"""Generate optimizer-specific gold data for the 3D strain regression tests.

Examples
--------
Generate the ICGN baselines used by the tests::

    python scripts/generate_strain_gold.py --optimizer ICGN

Regenerate the FALM baselines::

    python scripts/generate_strain_gold.py --optimizer FALM
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

# Match the deterministic single-threaded configuration used by the tests.
os.environ["OMP_NUM_THREADS"] = "1"

import pyvale.calib as calib
import pyvale.data as dataset
import pyvale.dic as dic
import pyvale.strain as strain


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "tests" / "strain"
GOLD_DIR = TEST_DIR / "gold"

GOLD_FIELDS = (
    "def_00",
    "def_01",
    "def_10",
    "def_11",
    "def_20",
    "def_21",
    "eps_xx",
    "eps_xy",
    "eps_yx",
    "eps_yy",
)


def generate_gold(optimizer: str, window_elements: list[int]) -> None:
    ref0 = dataset.dic_plate_with_hole_cam0_ref()
    ref1 = dataset.dic_plate_with_hole_cam1_ref()
    def0 = dataset.dic_plate_with_hole_cam0_def()
    def1 = dataset.dic_plate_with_hole_cam1_def()

    roi = dic.RegionOfInterest(ref0)
    roi.read_yaml(TEST_DIR / "roi.yaml")
    calibration = calib.loadtxt(TEST_DIR / "calib_hole.txt")

    with tempfile.TemporaryDirectory(prefix="pyvale_strain_gold_") as directory:
        output_path = Path(directory)
        dic.calculate_3d(
            reference=[ref0, ref1],
            deformed=[def0, def1],
            calibration=calibration,
            roi_mask=roi.mask,
            seed=roi.seed,
            subset_size=21,
            subset_step=10,
            max_displacement=10,
            output_basepath=output_path,
            output_prefix="dic_3d_",
            output_delimiter=",",
            image_filter_kernel=1,
            optimizer=optimizer,
            print_level=0,
        )

        for window_element in window_elements:
            strain.calculate_3d(
                data=output_path / "dic_3d_*csv",
                window_size=5,
                window_element=window_element,
                output_basepath=output_path,
                output_prefix="strain_3d_",
                strain_formulation="ALMANSI",
                partial_window=1,
                print_level=0,
            )
            result = strain.import_3d(output_path / "strain_3d_*")
            output_file = GOLD_DIR / (
                f"strain_3d_{optimizer.lower()}_q{window_element}.npz"
            )
            np.savez(
                output_file,
                **{
                    field: getattr(result, field)[1]
                    for field in GOLD_FIELDS
                },
            )
            print(f"Wrote {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimizer",
        choices=("FALM", "ICGN"),
        required=True,
        help="DIC optimizer used to generate the baseline.",
    )
    parser.add_argument(
        "--window-element",
        type=int,
        choices=(4, 9),
        nargs="+",
        default=[4, 9],
        dest="window_elements",
        help="Strain window elements to generate (default: 4 9).",
    )
    args = parser.parse_args()
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    generate_gold(args.optimizer, args.window_elements)


if __name__ == "__main__":
    main()
