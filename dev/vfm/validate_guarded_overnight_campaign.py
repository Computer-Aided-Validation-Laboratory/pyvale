"""Read-only preflight for the frozen five-run guarded campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyvale.vfm import ExperimentData, VfmRegionOfInterest
from pyvale.vfm.egisupports import generate_odd_pixel_egi_support_bank


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest_path = args.manifest or (
        root / "dev/vfm/data/wdbn1_guarded_overnight_campaign_v1_20260901.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    records: list[dict[str, object]] = []
    for case in manifest["runs"]:
        input_path = Path(case["input"])
        if case["dataset_kind"] == "experimental" and args.experimental_input is not None:
            input_path = args.experimental_input
        elif case["dataset_kind"] != "experimental" and args.synthetic_root is not None:
            marker = "/notched-ebw/synthetic-fe/"
            relative = str(input_path).split(marker, 1)[1]
            input_path = args.synthetic_root / relative
        experiment_path = input_path / "experiment_data.yaml"
        if not experiment_path.is_file():
            failures.append(f"{case['name']}: missing {experiment_path}")
            continue
        experiment = ExperimentData.load_from_file(experiment_path)
        x = experiment.specimen_geometry.x
        y = experiment.specimen_geometry.y
        mask_before = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(x, y)
        correction_enabled = case["fre_domain_correction"] == "enabled"
        if correction_enabled:
            roi_path = root / case["fre_region_of_interest"]
            if not roi_path.is_file():
                failures.append(f"{case['name']}: missing FRE ROI {roi_path}")
                continue
            experiment.specimen_geometry.force_reconstruction_region_of_interest = (
                VfmRegionOfInterest.from_yaml(roi_path)
            )
        mask_after = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(x, y)
        if not np.array_equal(mask_before, mask_after):
            failures.append(f"{case['name']}: FRE correction changed measured EGI ROI")
        bank = generate_odd_pixel_egi_support_bank(x, y)
        fine_size = int(case["fine_egi_window"])
        by_size = {support.window_size[0]: support for support in bank}
        if fine_size not in by_size:
            failures.append(f"{case['name']}: fine support {fine_size} is outside geometry bank")
            continue
        resolved = {
            "fine": list(by_size[fine_size].window_size),
            "broad": list(bank[-1].window_size),
        }
        expected_broad = int(case["expected_broad_egi_window"])
        if resolved["broad"] != [expected_broad, expected_broad]:
            failures.append(
                f"{case['name']}: expected broad {expected_broad}, got {resolved['broad']}"
            )
        records.append({
            "name": case["name"],
            "grid_shape": list(x.shape),
            "fine_window": resolved["fine"],
            "broad_window": resolved["broad"],
            "fre_domain_correction": case["fre_domain_correction"],
            "measured_roi_points": int(np.count_nonzero(mask_before)),
            "known_truth_available": (input_path / "known_parameter_maps.npz").is_file(),
        })
    payload = {"status": "PASS" if not failures else "FAIL", "runs": records, "failures": failures}
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--synthetic-root", type=Path)
    parser.add_argument("--experimental-input", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
