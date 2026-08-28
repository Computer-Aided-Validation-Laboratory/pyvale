"""Extract a compact strain-noise model from unloaded WDBN1 DIC frames.

Run with an environment containing h5py (the dic-processing-tools environment
is suitable).  The output JSON contains no experimental field data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


COMPONENTS = ("exx", "eyy", "exy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assembled-h5", type=Path, required=True)
    parser.add_argument("--force-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-frames", default="0:11")
    parser.add_argument("--rows", default="486:1016")
    args = parser.parse_args()
    frames = _slice(args.static_frames)
    rows = _slice(args.rows)

    with h5py.File(args.assembled_h5, "r") as archive:
        mask = np.asarray(
            archive["measurement_valid_mask"][frames, rows, :]
        ).all(axis=0)
        x = np.asarray(archive["x_ref"][rows, :], dtype=float)
        y = np.asarray(archive["y_ref"][rows, :], dtype=float)
        data = np.stack([
            np.asarray(archive[name][frames, rows, :], dtype=float)
            for name in COMPONENTS
        ], axis=1)

    dx = float(np.nanmedian(np.abs(np.diff(x, axis=1))))
    dy = float(np.nanmedian(np.abs(np.diff(y, axis=0))))
    valid = np.broadcast_to(mask, (data.shape[0], 3, *mask.shape))
    static_mean = np.zeros_like(data[0])
    static_mean[:, mask] = np.mean(data[:, :, mask], axis=0)
    centred = np.where(valid, data-static_mean, np.nan)
    flattened = np.moveaxis(centred, 1, -1)[
        np.broadcast_to(mask, (data.shape[0], *mask.shape))
    ]
    model = {
        "schema": "pyvale-wdbn1-noise-v1",
        "source": str(args.assembled_h5.resolve()),
        "static_frames": [frames.start, frames.stop],
        "grid_spacing_mm": {"x": dx, "y": dy},
        "component_correlation": np.corrcoef(flattened, rowvar=False).tolist(),
        "components": {},
        "notes": [
            "Temporal pointwise noise after removal of each point's static mean.",
            "Correlation lengths are first positive-lag crossings below exp(-1).",
            "Gaussian filter sigma is half that correlation length.",
        ],
    }
    differences = np.diff(data, axis=0) / np.sqrt(2.0)
    for index, name in enumerate(COMPONENTS):
        point_sigma = np.full(mask.shape, np.nan)
        point_sigma[mask] = np.std(
            centred[:, index, mask], axis=0, ddof=1,
        )
        field = np.where(mask, differences[:, index], np.nan)
        length_x = _correlation_length(field, axis=2, spacing=dx)
        length_y = _correlation_length(field, axis=1, spacing=dy)
        model["components"][name] = {
            "sigma": float(np.sqrt(np.nanmean(point_sigma**2))),
            "sigma_microstrain": float(np.sqrt(np.nanmean(point_sigma**2))*1e6),
            "correlation_length_mm": {"x": length_x, "y": length_y},
            "gaussian_filter_sigma_mm": {"x": .5*length_x, "y": .5*length_y},
        }
    forces = []
    with args.force_history.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["image_number"]) >= frames.stop:
                break
            forces.append(float(row["force_N"]))
    model["force"] = {
        "mean_n": float(np.mean(forces)),
        "sigma_n": float(np.std(forces, ddof=1)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(json.dumps(model, indent=2))


def _slice(value: str) -> slice:
    start, stop = value.split(":", 1)
    return slice(int(start), int(stop))


def _correlation_length(field, *, axis: int, spacing: float) -> float:
    variance = float(np.nanmean(field*field))
    threshold = np.exp(-1.0)
    for lag in range(1, min(field.shape[axis]//2, 100)):
        left = [slice(None)] * field.ndim
        right = [slice(None)] * field.ndim
        left[axis] = slice(None, -lag)
        right[axis] = slice(lag, None)
        correlation = float(np.nanmean(field[tuple(left)]*field[tuple(right)])/variance)
        if correlation <= threshold:
            return lag * spacing
    return 100 * spacing


if __name__ == "__main__":
    main()
