from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_VFM_DIR = REPO_ROOT / "tests" / "vfm"
if str(TESTS_VFM_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_VFM_DIR))

from slicewise_validation_case import (
    DEFAULT_VFM_TEST_DATA_ROOT,
    ensure_rectangle_slicewise_yield_case,
)


def main() -> None:
    args = _parse_args()
    processed_case = ensure_rectangle_slicewise_yield_case(
        args.data_root,
        overwrite=args.overwrite,
    )
    print(f"Case root: {processed_case.case_root}")
    print(f"Raw Ansys-style export: {processed_case.raw_export.raw_dir}")
    print(f"Prepared experiment data: {processed_case.experiment_data_file}")
    print(f"Known parameter maps: {processed_case.known_parameter_maps_file}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the rectangle slice-wise yield-strength VFM test fixture."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_VFM_TEST_DATA_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate raw and prepared fixture folders even when they already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
