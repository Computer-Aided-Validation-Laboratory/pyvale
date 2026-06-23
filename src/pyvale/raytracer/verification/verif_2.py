#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import re
import sys

import numpy as np


FRAME_RE = re.compile(r"cam\d+_frame(\d+)_field\d+_stats\.csv$")

GAUSS_POINTS = np.array(
    [
        -math.sqrt(3.0 / 5.0),
        0.0,
        math.sqrt(3.0 / 5.0),
    ],
    dtype=float,
)
GAUSS_WEIGHTS = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verif-root",
        type=pathlib.Path,
        default=repo_root() / "verification" / "verif" / "verif_2",
    )
    parser.add_argument(
        "--out-csv",
        type=pathlib.Path,
        default=repo_root() / "verification" / "verif" / "verif_2" / "verif_2_summary.csv",
    )
    parser.add_argument(
        "--subset",
        choices=("high_order_all", "high_order_bulge_tan", "all"),
        default="all",
    )
    parser.add_argument(
        "--edge-segments",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--edge-segments-sensor-mult",
        type=int,
        default=1,
    )
    return parser.parse_args()


def parse_stats_csv(stats_path: pathlib.Path) -> dict[str, float]:
    stats: dict[str, float] = {}
    with stats_path.open(newline="") as stats_file:
        reader = csv.reader(stats_file)
        header = next(reader)
        if header != ["key", "unit", "value"]:
            raise ValueError(
                f"{stats_path} does not have key,unit,value layout",
            )
        for row in reader:
            if len(row) != 3:
                continue
            key, _, value = row
            stats[key] = float(value)
    return stats


def load_scalar_csv(image_path: pathlib.Path) -> np.ndarray:
    rows: list[list[float]] = []
    with image_path.open(newline="") as image_file:
        reader = csv.reader(image_file)
        for row in reader:
            if row and row[-1] == "":
                row = row[:-1]
            rows.append([float(value) for value in row])
    return np.asarray(rows, dtype=float)


def calc_image_centroid_and_area(
    image_vals: np.ndarray,
) -> tuple[float, float, float, float]:
    rows_num, cols_num = image_vals.shape
    fill_value = float(np.max(image_vals))
    weighted_area = float(np.sum(image_vals))
    if weighted_area <= 0.0 or fill_value <= 0.0:
        return math.nan, math.nan, 0.0, fill_value

    x_centers = np.arange(cols_num, dtype=float) + 0.5
    y_centers = np.arange(rows_num, dtype=float) + 0.5

    centroid_x = float(np.sum(image_vals * x_centers[None, :]) / weighted_area)
    centroid_y = float(np.sum(image_vals * y_centers[:, None]) / weighted_area)
    area_px2 = weighted_area / fill_value
    return centroid_x, centroid_y, area_px2, fill_value


def parse_case_info(stats_path: pathlib.Path) -> tuple[str, str, str, int]:
    case_name = stats_path.parent.name
    case_parts = case_name.split("_")
    mesh_name = case_parts[1]
    if mesh_name == "quad4ibi" or mesh_name == "quad4newton":
        mesh_name = "quad4"
    distort_name = case_parts[2]

    frame_match = FRAME_RE.match(stats_path.name)
    if frame_match is None:
        raise ValueError(f"could not parse frame index from {stats_path.name}")

    return case_name, mesh_name, distort_name, int(frame_match.group(1))


def get_node_coords(stats: dict[str, float]) -> list[tuple[float, float]]:
    node_indices: list[int] = []
    for key in stats:
        if key.startswith("N") and key.endswith("_x"):
            node_indices.append(int(key[1:-2]))

    coords = []
    for nn in sorted(node_indices):
        coords.append((stats[f"N{nn}_x"], stats[f"N{nn}_y"]))
    return coords


def linear_edge_nodes(
    node_start: tuple[float, float],
    node_end: tuple[float, float],
    ss: float,
) -> tuple[float, float, float, float]:
    n0 = 0.5 * (1.0 - ss)
    n1 = 0.5 * (1.0 + ss)
    dn0 = -0.5
    dn1 = 0.5

    xx = n0 * node_start[0] + n1 * node_end[0]
    yy = n0 * node_start[1] + n1 * node_end[1]
    dx_ds = dn0 * node_start[0] + dn1 * node_end[0]
    dy_ds = dn0 * node_start[1] + dn1 * node_end[1]
    return xx, yy, dx_ds, dy_ds


def quadratic_edge_nodes(
    node_start: tuple[float, float],
    node_mid: tuple[float, float],
    node_end: tuple[float, float],
    ss: float,
) -> tuple[float, float, float, float]:
    n0 = 0.5 * ss * (ss - 1.0)
    n1 = 1.0 - ss * ss
    n2 = 0.5 * ss * (ss + 1.0)
    dn0 = ss - 0.5
    dn1 = -2.0 * ss
    dn2 = ss + 0.5

    xx = n0 * node_start[0] + n1 * node_mid[0] + n2 * node_end[0]
    yy = n0 * node_start[1] + n1 * node_mid[1] + n2 * node_end[1]
    dx_ds = dn0 * node_start[0] + dn1 * node_mid[0] + dn2 * node_end[0]
    dy_ds = dn0 * node_start[1] + dn1 * node_mid[1] + dn2 * node_end[1]
    return xx, yy, dx_ds, dy_ds


def iter_edges(
    mesh_name: str,
    node_coords: list[tuple[float, float]],
) -> list[tuple[str, tuple[float, float], ...]]:
    print(mesh_name)
    if mesh_name == "TRI3":
        return [
            ("linear", node_coords[0], node_coords[1]),
            ("linear", node_coords[1], node_coords[2]),
            ("linear", node_coords[2], node_coords[0]),
        ]
    if mesh_name == "TRI6":
        return [
            ("quadratic", node_coords[0], node_coords[3], node_coords[1]),
            ("quadratic", node_coords[1], node_coords[4], node_coords[2]),
            ("quadratic", node_coords[2], node_coords[5], node_coords[0]),
        ]
    if mesh_name == "QUAD4":
        return [
            ("linear", node_coords[0], node_coords[1]),
            ("linear", node_coords[1], node_coords[2]),
            ("linear", node_coords[2], node_coords[3]),
            ("linear", node_coords[3], node_coords[0]),
        ]
    if mesh_name == "QUAD8":
        return [
            ("quadratic", node_coords[0], node_coords[4], node_coords[1]),
            ("quadratic", node_coords[1], node_coords[5], node_coords[2]),
            ("quadratic", node_coords[2], node_coords[6], node_coords[3]),
            ("quadratic", node_coords[3], node_coords[7], node_coords[0]),
        ]
    if mesh_name == "QUAD9":
        return [
            ("quadratic", node_coords[0], node_coords[4], node_coords[1]),
            ("quadratic", node_coords[1], node_coords[5], node_coords[2]),
            ("quadratic", node_coords[2], node_coords[6], node_coords[3]),
            ("quadratic", node_coords[3], node_coords[7], node_coords[0]),
        ]
    raise ValueError(f"unsupported mesh name {mesh_name}")


def calc_edge_area(
    edge_kind: str,
    edge_nodes: tuple[tuple[float, float], ...],
) -> float:
    edge_area = 0.0
    for ss, ww in zip(GAUSS_POINTS, GAUSS_WEIGHTS, strict=True):
        if edge_kind == "linear":
            xx, yy, dx_ds, dy_ds = linear_edge_nodes(
                edge_nodes[0],
                edge_nodes[1],
                float(ss),
            )
        else:
            xx, yy, dx_ds, dy_ds = quadratic_edge_nodes(
                edge_nodes[0],
                edge_nodes[1],
                edge_nodes[2],
                float(ss),
            )
        edge_area += ww * (xx * dy_ds - yy * dx_ds)
    return 0.5 * edge_area


def calc_reference_area_px2(
    mesh_name: str,
    node_coords: list[tuple[float, float]],
) -> float:
    signed_area = 0.0
    for edge in iter_edges(mesh_name, node_coords):
        signed_area += calc_edge_area(edge[0], edge[1:])
    return abs(signed_area)


def sample_edge_points(
    edge_kind: str,
    edge_nodes: tuple[tuple[float, float], ...],
    points_num: int,
) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    for nn in range(points_num + 1):
        ss = -1.0 + 2.0 * float(nn) / float(points_num)
        if edge_kind == "linear":
            xx, yy, _, _ = linear_edge_nodes(
                edge_nodes[0],
                edge_nodes[1],
                ss,
            )
        else:
            xx, yy, _, _ = quadratic_edge_nodes(
                edge_nodes[0],
                edge_nodes[1],
                edge_nodes[2],
                ss,
            )
        samples.append((xx, yy))
    return samples


def build_boundary_polyline(
    mesh_name: str,
    node_coords: list[tuple[float, float]],
    edge_segments: int,
) -> list[tuple[float, float]]:
    polyline: list[tuple[float, float]] = []
    for edge_idx, edge in enumerate(iter_edges(mesh_name, node_coords)):
        edge_points = sample_edge_points(edge[0], edge[1:], edge_segments)
        if edge_idx > 0:
            edge_points = edge_points[1:]
        polyline.extend(edge_points)
    return polyline


def calc_edge_segments(
    sensor_x: int,
    sensor_y: int,
    edge_segments: int | None,
    edge_segments_sensor_mult: int,
) -> int:
    if edge_segments is not None:
        return edge_segments
    sensor_edge_max = max(sensor_x, sensor_y)
    return max(1024, edge_segments_sensor_mult * sensor_edge_max)


def build_reference_mask(
    mesh_name: str,
    node_coords: list[tuple[float, float]],
    sensor_x: int,
    sensor_y: int,
    edge_segments: int,
    eps: float = 1.0e-12,
) -> np.ndarray:
    polyline = build_boundary_polyline(mesh_name, node_coords, edge_segments)
    if len(polyline) < 2:
        raise ValueError("reference polyline must have at least 2 points")

    xx_centers = np.arange(sensor_x, dtype=float) + 0.5
    mask = np.zeros((sensor_y, sensor_x), dtype=float)

    seg_x0 = np.array([point[0] for point in polyline], dtype=float)
    seg_y0 = np.array([point[1] for point in polyline], dtype=float)
    seg_x1 = np.roll(seg_x0, -1)
    seg_y1 = np.roll(seg_y0, -1)

    for rr in range(sensor_y):
        yy = float(rr) + 0.5
        seg_dy = seg_y1 - seg_y0
        non_horizontal_mask = np.abs(seg_dy) > eps
        upward_mask = (
            (seg_y0 <= yy + eps) &
            (yy < seg_y1 - eps) &
            non_horizontal_mask
        )
        downward_mask = (
            (seg_y1 <= yy + eps) &
            (yy < seg_y0 - eps) &
            non_horizontal_mask
        )
        crossing_mask = upward_mask | downward_mask

        row_mask = np.zeros(sensor_x, dtype=bool)
        if np.any(crossing_mask):
            event_x_arr = seg_x0[crossing_mask] + (
                (yy - seg_y0[crossing_mask]) *
                (seg_x1[crossing_mask] - seg_x0[crossing_mask]) /
                seg_dy[crossing_mask]
            )
            event_winding_arr = np.where(
                upward_mask[crossing_mask],
                1,
                -1,
            ).astype(int)
            order = np.argsort(event_x_arr)
            event_x_arr = event_x_arr[order]
            event_winding_arr = event_winding_arr[order]
            winding_prefix = np.cumsum(event_winding_arr, dtype=int)
            event_idx = np.searchsorted(
                event_x_arr,
                xx_centers,
                side="right",
            ) - 1

            inside_mask = event_idx >= 0
            row_winding = np.zeros(sensor_x, dtype=int)
            row_winding[inside_mask] = winding_prefix[event_idx[inside_mask]]
            row_mask = row_winding != 0

        # Treat pixel centers that lie exactly on the discretized boundary
        # as inside so the reference mask matches the rasterizer's boundary
        # inclusion convention on sensor-aligned cases.
        horizontal_on_row_mask = (
            (np.abs(seg_y0 - yy) <= eps) &
            (np.abs(seg_y1 - yy) <= eps)
        )
        if np.any(horizontal_on_row_mask):
            x_min_arr = np.minimum(
                seg_x0[horizontal_on_row_mask],
                seg_x1[horizontal_on_row_mask],
            )
            x_max_arr = np.maximum(
                seg_x0[horizontal_on_row_mask],
                seg_x1[horizontal_on_row_mask],
            )
            for x_min, x_max in zip(x_min_arr, x_max_arr, strict=True):
                row_mask |= (
                    (xx_centers >= x_min - eps) &
                    (xx_centers <= x_max + eps)
                )

        touching_row_mask = (
            (yy >= np.minimum(seg_y0, seg_y1) - eps) &
            (yy <= np.maximum(seg_y0, seg_y1) + eps) &
            non_horizontal_mask
        )
        if np.any(touching_row_mask):
            x_on_seg_arr = seg_x0[touching_row_mask] + (
                (yy - seg_y0[touching_row_mask]) *
                (seg_x1[touching_row_mask] - seg_x0[touching_row_mask]) /
                seg_dy[touching_row_mask]
            )
            for x_on_seg in x_on_seg_arr:
                xx_idx = int(round(x_on_seg - 0.5))
                if 0 <= xx_idx < sensor_x:
                    xx_center = xx_centers[xx_idx]
                    if abs(xx_center - x_on_seg) <= eps:
                        row_mask[xx_idx] = True

        mask[rr, row_mask] = 1.0

    return mask


def normalize_render_mask(
    image_vals: np.ndarray,
) -> np.ndarray:
    fill_value = float(np.max(image_vals))
    if fill_value <= 0.0:
        return np.zeros_like(image_vals)
    return image_vals / fill_value


def calc_mask_diff_pct(
    mask_ref: np.ndarray,
    mask_render: np.ndarray,
) -> float:
    abs_diff = np.abs(mask_ref - mask_render)
    return 100.0 * float(np.mean(abs_diff))


def run_self_test() -> None:
    side_leng = 10.0
    tri_height = 0.5 * math.sqrt(3.0) * side_leng
    tri3_nodes = [
        (0.0, 0.0),
        (side_leng, 0.0),
        (0.5 * side_leng, tri_height),
    ]
    tri6_nodes = [
        tri3_nodes[0],
        tri3_nodes[1],
        tri3_nodes[2],
        (0.5 * side_leng, 0.0),
        (0.75 * side_leng, 0.5 * tri_height),
        (0.25 * side_leng, 0.5 * tri_height),
    ]
    area_exact = 0.25 * math.sqrt(3.0) * side_leng * side_leng

    tri3_area = calc_reference_area_px2("TRI3", tri3_nodes)
    tri6_area = calc_reference_area_px2("TRI6", tri6_nodes)

    if not math.isclose(tri3_area, area_exact, rel_tol=0.0, abs_tol=1.0e-10):
        raise RuntimeError("tri3 area self-test failed")
    if not math.isclose(tri6_area, area_exact, rel_tol=0.0, abs_tol=1.0e-10):
        raise RuntimeError("tri6 area self-test failed")


def analyse_stats_file(
    stats_path: pathlib.Path,
    edge_segments: int | None,
    edge_segments_sensor_mult: int,
) -> dict[str, object]:
    stats = parse_stats_csv(stats_path)
    image_path = stats_path.with_name(stats_path.name.replace("_stats.csv", ".csv"))
    image_vals = load_scalar_csv(image_path)

    case_name, mesh_name, distort_name, frame_idx = parse_case_info(stats_path)
    centroid_num_x, centroid_num_y, area_num_px2, fill_value = calc_image_centroid_and_area(
        image_vals,
    )

    # node_coords = get_node_coords(stats)
    # area_ref_px2 = calc_reference_area_px2(mesh_name, node_coords)
    if "sensor_pixels_x" in stats and "sensor_pixels_y" in stats:
        sensor_x = int(round(stats["sensor_pixels_x"]))
        sensor_y = int(round(stats["sensor_pixels_y"]))
    else:
        sensor_y, sensor_x = image_vals.shape
    edge_segments_num = calc_edge_segments(
        sensor_x,
        sensor_y,
        edge_segments,
        edge_segments_sensor_mult,
    )
    # mask_ref = build_reference_mask(
    #     mesh_name,
    #     node_coords,
    #     sensor_x,
    #     sensor_y,
    #     edge_segments_num,
    # )
    mask_render = normalize_render_mask(image_vals)
    # if mask_render.shape != mask_ref.shape:
    #     raise ValueError(
    #         f"mask shape mismatch for {stats_path}: "
    #         f"{mask_ref.shape} vs {mask_render.shape}",
    #     )
    # mask_diff_pct = calc_mask_diff_pct(mask_ref, mask_render)

    centroid_ref_x = stats["cent_ideal_x"]
    centroid_ref_y = stats["cent_ideal_y"]
    centroid_diff_x = centroid_num_x - centroid_ref_x
    centroid_diff_y = centroid_num_y - centroid_ref_y
    centroid_dist = math.hypot(centroid_diff_x, centroid_diff_y)

    # area_diff_px2 = area_num_px2 - area_ref_px2
    # area_rel_diff = area_diff_px2 / area_ref_px2 if area_ref_px2 != 0.0 else math.nan
    # area_err_pct = 100.0 * area_rel_diff

    return {
        "case_name": case_name,
        "mesh_name": mesh_name,
        "distort_name": distort_name,
        "frame_idx": frame_idx,
        # "node_count": len(node_coords),
        # "node_count": 0.0,
        "fill_value": fill_value,
        "centroid_ref_x_px": centroid_ref_x,
        "centroid_ref_y_px": centroid_ref_y,
        "centroid_num_x_px": centroid_num_x,
        "centroid_num_y_px": centroid_num_y,
        "centroid_diff_x_px": centroid_diff_x,
        "centroid_diff_y_px": centroid_diff_y,
        "centroid_dist_px": centroid_dist,
        # "area_ref_px2": area_ref_px2,
        # "area_ref_px2": 0.0,
        "area_num_px2": area_num_px2,
        # "area_diff_px2": area_diff_px2,
        # "area_rel_diff": area_rel_diff,
        # "area_err_pct": area_err_pct,
        # "area_diff_px2": 0.0,
        # "area_rel_diff": 0.0,
        # "area_err_pct": 0.0,
        "sensor_x_px": sensor_x,
        "sensor_y_px": sensor_y,
        "edge_segments": edge_segments_num,
        # "mask_diff_pct": mask_diff_pct,
        # "mask_diff_pct": 0.0,
        "stats_path": str(stats_path.relative_to(repo_root())),
        "image_path": str(image_path.relative_to(repo_root())),
    }


def write_summary_csv(
    out_csv: pathlib.Path,
    rows: list[dict[str, object]],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    field_names = [
        "case_name",
        "mesh_name",
        "distort_name",
        "frame_idx",
        # "node_count",
        "fill_value",
        "centroid_ref_x_px",
        "centroid_ref_y_px",
        "centroid_num_x_px",
        "centroid_num_y_px",
        "centroid_diff_x_px",
        "centroid_diff_y_px",
        "centroid_dist_px",
        # "area_ref_px2",
        "area_num_px2",
        # "area_diff_px2",
        # "area_rel_diff",
        # "area_err_pct",
        "sensor_x_px",
        "sensor_y_px",
        "edge_segments",
        # "mask_diff_pct",
        "stats_path",
        "image_path",
    ]
    with out_csv.open("w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def keep_row(row: dict[str, object], subset: str) -> bool:
    if subset == "all":
        return True

    mesh_name = str(row["mesh_name"])
    distort_name = str(row["distort_name"])
    return keep_case_info(mesh_name, distort_name, subset)


def keep_case_info(mesh_name: str, distort_name: str, subset: str) -> bool:
    if subset == "all":
        return True

    if subset == "high_order_bulge_tan":
        return mesh_name in {"tri6", "quad8", "quad9"} and distort_name in {
            "bulge",
            "tan",
        }

    return mesh_name in {"tri6", "quad8", "quad9"} and distort_name in {
        "bulge",
        "tan",
        "stretch",
        "shear",
        "rot",
    }


def main() -> int:
    args = parse_args()
    print("Running verif_2 self-test...")
    run_self_test()
    print("Self-test passed.")

    stats_files_all = sorted(args.verif_root.glob("b_*/*_stats.csv"))
    print(f"Found {len(stats_files_all)} total verif_2 stats files.")
    stats_files = []
    for stats_path in stats_files_all:
        _, mesh_name, distort_name, _ = parse_case_info(stats_path)
        if keep_case_info(mesh_name, distort_name, args.subset):
            stats_files.append(stats_path)
    if not stats_files:
        raise FileNotFoundError(f"no stats files found under {args.verif_root}")
    print(
        f"Selected {len(stats_files)} stats files for subset "
        f"'{args.subset}'.",
    )

    rows = []
    for file_idx, stats_path in enumerate(stats_files, start=1):
        if file_idx == 1 or file_idx % 10 == 0 or file_idx == len(stats_files):
            print(
                f"Analysing file {file_idx}/{len(stats_files)}: "
                f"{stats_path.parent.name}/{stats_path.name}",
            )
        rows.append(analyse_stats_file(
            stats_path,
            args.edge_segments,
            args.edge_segments_sensor_mult,
        ))
    rows.sort(key=lambda row: (str(row["case_name"]), int(row["frame_idx"])))
    print(f"Writing summary CSV to {args.out_csv}...")
    write_summary_csv(args.out_csv, rows)

    max_centroid = max(abs(float(row["centroid_dist_px"])) for row in rows)
    # max_area_rel = max(abs(float(row["area_rel_diff"])) for row in rows)
    # max_mask_diff = max(abs(float(row["mask_diff_pct"])) for row in rows)
    print(f"Processed {len(rows)} verif_2 frames")
    print(f"Wrote summary to {args.out_csv}")
    print(f"Max centroid error: {max_centroid:.6f} px")
    # print(f"Max area rel error: {max_area_rel:.6e}")
    # print(f"Max mask diff: {max_mask_diff:.6f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
