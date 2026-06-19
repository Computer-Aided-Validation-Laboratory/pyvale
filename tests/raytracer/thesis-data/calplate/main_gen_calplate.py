from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


BASE_DIR = Path(__file__).resolve().parent

TEXTURE_PIXELS_X = 2660
TEXTURE_PIXELS_Y = 1774
ASPECT_RATIO = TEXTURE_PIXELS_X / TEXTURE_PIXELS_Y

# Target sizes:
# Target, 150x100,   dot spacing = 10 mm
# Target, 75x50,     dot spacing = 5 mm
# Target, 37.5x25,   dot spacing = 2.5 mm
# Target, 18.75x12.5 dot spacing = 1.25 mm

L = 18.75e-3
SHORT_DIM = L / ASPECT_RATIO

CAL_MODE = "STRAT" # RANDOM, STRAT, FULL
CAL_RANDOM_IMAGES = 100
CAL_STRAT_IMAGES = 100
CAL_RANDOM_SEED = 7

X_TRANS_LIMITS = (-0.2, 0.2)
Y_TRANS_LIMITS = (-0.1, 0.1)
Z_TRANS_LIMITS = (-0.2, 0.2)

X_TRANS_STEP = 0.05
Y_TRANS_STEP = 0.05
Z_TRANS_STEP = 0.05

X_ROT_LIMITS = (-10.0, 10.0)
Y_ROT_LIMITS = (-10.0, 10.0)
Z_ROT_LIMITS = (-10.0, 10.0)

X_ROT_STEP = 1.0
Y_ROT_STEP = 1.0
Z_ROT_STEP = 1.0


@dataclass(frozen=True)
class MotionState:
    x_trans_mul: float
    y_trans_mul: float
    z_trans_mul: float
    x_rot_deg: float
    y_rot_deg: float
    z_rot_deg: float


ZERO_STATE = MotionState(
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


def axis_values(limits: tuple[float, float], step: float) -> np.ndarray:
    lower, upper = limits
    if step <= 0.0:
        raise ValueError("Axis step must be positive.")
    if upper < lower:
        raise ValueError("Axis upper limit must be >= lower limit.")

    values = np.arange(lower, upper + 0.5 * step, step, dtype=np.float64)
    values = np.clip(values, lower, upper)
    values = np.round(values, 10)
    values = np.unique(values)
    return values


def corner_uvs() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )


def centered_plate_corners() -> np.ndarray:
    half_length = 0.5 * L
    half_short = 0.5 * SHORT_DIM
    return np.array(
        [
            [-half_length, -half_short, 0.0],
            [half_length, -half_short, 0.0],
            [half_length, half_short, 0.0],
            [-half_length, half_short, 0.0],
        ],
        dtype=np.float64,
    )


def midpoint(point_aa: np.ndarray, point_bb: np.ndarray) -> np.ndarray:
    return 0.5 * (point_aa + point_bb)


def midpoint_uv(uv_aa: np.ndarray, uv_bb: np.ndarray) -> np.ndarray:
    return 0.5 * (uv_aa + uv_bb)


def tri3_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = centered_plate_corners()
    connect = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.int64,
    )
    uvs = corner_uvs()
    return coords, connect, uvs


def tri6_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners = centered_plate_corners()
    corner_uv = corner_uvs()

    mid_01 = midpoint(corners[0], corners[1])
    mid_12 = midpoint(corners[1], corners[2])
    mid_23 = midpoint(corners[2], corners[3])
    mid_30 = midpoint(corners[3], corners[0])
    mid_02 = midpoint(corners[0], corners[2])

    uv_01 = midpoint_uv(corner_uv[0], corner_uv[1])
    uv_12 = midpoint_uv(corner_uv[1], corner_uv[2])
    uv_23 = midpoint_uv(corner_uv[2], corner_uv[3])
    uv_30 = midpoint_uv(corner_uv[3], corner_uv[0])
    uv_02 = midpoint_uv(corner_uv[0], corner_uv[2])

    coords = np.vstack(
        [
            corners,
            mid_01,
            mid_12,
            mid_23,
            mid_30,
            mid_02,
        ]
    )
    uvs = np.vstack(
        [
            corner_uv,
            uv_01,
            uv_12,
            uv_23,
            uv_30,
            uv_02,
        ]
    )
    connect = np.array(
        [
            [0, 1, 2, 4, 5, 8],
            [0, 2, 3, 8, 6, 7],
        ],
        dtype=np.int64,
    )
    return coords, connect, uvs


def quad4_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = centered_plate_corners()
    connect = np.array([[0, 1, 2, 3]], dtype=np.int64)
    uvs = corner_uvs()
    return coords, connect, uvs


def quad8_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners = centered_plate_corners()
    corner_uv = corner_uvs()

    mid_01 = midpoint(corners[0], corners[1])
    mid_12 = midpoint(corners[1], corners[2])
    mid_23 = midpoint(corners[2], corners[3])
    mid_30 = midpoint(corners[3], corners[0])

    uv_01 = midpoint_uv(corner_uv[0], corner_uv[1])
    uv_12 = midpoint_uv(corner_uv[1], corner_uv[2])
    uv_23 = midpoint_uv(corner_uv[2], corner_uv[3])
    uv_30 = midpoint_uv(corner_uv[3], corner_uv[0])

    coords = np.vstack([corners, mid_01, mid_12, mid_23, mid_30])
    uvs = np.vstack([corner_uv, uv_01, uv_12, uv_23, uv_30])
    connect = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    return coords, connect, uvs


def quad9_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quad8_coords, quad8_connect, quad8_uvs = quad8_mesh()
    center = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    center_uv = np.array([[0.5, 0.5]], dtype=np.float64)
    coords = np.vstack([quad8_coords, center])
    uvs = np.vstack([quad8_uvs, center_uv])
    connect = np.array(
        [[
            quad8_connect[0, 0],
            quad8_connect[0, 1],
            quad8_connect[0, 2],
            quad8_connect[0, 3],
            quad8_connect[0, 4],
            quad8_connect[0, 5],
            quad8_connect[0, 6],
            quad8_connect[0, 7],
            8,
        ]],
        dtype=np.int64,
    )
    return coords, connect, uvs


def mesh_cases() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        "tri3_calplate": tri3_mesh(),
        "tri6_calplate": tri6_mesh(),
        "quad4_calplate": quad4_mesh(),
        "quad8_calplate": quad8_mesh(),
        "quad9_calplate": quad9_mesh(),
    }


def all_motion_states() -> list[MotionState]:
    x_trans_values = axis_values(X_TRANS_LIMITS, X_TRANS_STEP)
    y_trans_values = axis_values(Y_TRANS_LIMITS, Y_TRANS_STEP)
    z_trans_values = axis_values(Z_TRANS_LIMITS, Z_TRANS_STEP)
    x_rot_values = axis_values(X_ROT_LIMITS, X_ROT_STEP)
    y_rot_values = axis_values(Y_ROT_LIMITS, Y_ROT_STEP)
    z_rot_values = axis_values(Z_ROT_LIMITS, Z_ROT_STEP)

    states = []
    for state_values in itertools.product(
        x_trans_values,
        y_trans_values,
        z_trans_values,
        x_rot_values,
        y_rot_values,
        z_rot_values,
    ):
        states.append(MotionState(*state_values))
    return states


def random_motion_states(
    states_full: list[MotionState],
    random_images: int,
    rng: np.random.Generator,
) -> list[MotionState]:
    if random_images > len(states_full):
        raise ValueError("Requested more random images than unique states.")
    sample_idx = rng.choice(len(states_full), size=random_images, replace=False)
    sample_idx.sort()
    return [states_full[ii] for ii in sample_idx]


def stratified_motion_states(
    states_full: list[MotionState],
    random_images: int,
    rng: np.random.Generator,
) -> list[MotionState]:
    if random_images > len(states_full):
        raise ValueError("Requested more stratified images than unique states.")

    x_trans_values = axis_values(X_TRANS_LIMITS, X_TRANS_STEP)
    y_trans_values = axis_values(Y_TRANS_LIMITS, Y_TRANS_STEP)
    z_trans_values = axis_values(Z_TRANS_LIMITS, Z_TRANS_STEP)
    x_rot_values = axis_values(X_ROT_LIMITS, X_ROT_STEP)
    y_rot_values = axis_values(Y_ROT_LIMITS, Y_ROT_STEP)
    z_rot_values = axis_values(Z_ROT_LIMITS, Z_ROT_STEP)
    axes = [
        x_trans_values,
        y_trans_values,
        z_trans_values,
        x_rot_values,
        y_rot_values,
        z_rot_values,
    ]

    selected_keys: set[tuple[float, ...]] = set()
    selected_states: list[MotionState] = []

    engine = qmc.LatinHypercube(d=6, seed=CAL_RANDOM_SEED)
    attempts = 0
    batch_size = max(random_images, 32)

    while len(selected_states) < random_images and attempts < 32:
        lhs_points = engine.random(batch_size)
        for point in lhs_points:
            state_values = []
            for axis_values_curr, sample in zip(axes, point, strict=True):
                axis_idx = min(
                    int(np.floor(sample * len(axis_values_curr))),
                    len(axis_values_curr) - 1,
                )
                state_values.append(float(axis_values_curr[axis_idx]))

            state_key = tuple(state_values)
            if state_key in selected_keys:
                continue

            selected_keys.add(state_key)
            selected_states.append(MotionState(*state_values))
            if len(selected_states) == random_images:
                break
        attempts += 1

    if len(selected_states) < random_images:
        states_remaining = [
            state
            for state in states_full
            if (
                state.x_trans_mul,
                state.y_trans_mul,
                state.z_trans_mul,
                state.x_rot_deg,
                state.y_rot_deg,
                state.z_rot_deg,
            )
            not in selected_keys
        ]
        fill_count = random_images - len(selected_states)
        fill_idx = rng.choice(
            len(states_remaining),
            size=fill_count,
            replace=False,
        )
        fill_idx.sort()
        selected_states.extend(states_remaining[ii] for ii in fill_idx)

    return selected_states


def selected_motion_states() -> list[MotionState]:
    states_full = all_motion_states()
    rng = np.random.default_rng(CAL_RANDOM_SEED)

    if CAL_MODE == "FULL":
        states_nonzero = [state for state in states_full if state != ZERO_STATE]
        return [ZERO_STATE, *states_nonzero]
    if CAL_MODE == "RANDOM":
        states_nonzero = [state for state in states_full if state != ZERO_STATE]
        states_random = random_motion_states(
            states_nonzero,
            CAL_RANDOM_IMAGES - 1,
            rng,
        )
        return [ZERO_STATE, *states_random]
    if CAL_MODE == "STRAT":
        states_nonzero = [state for state in states_full if state != ZERO_STATE]
        states_strat = stratified_motion_states(
            states_nonzero,
            CAL_STRAT_IMAGES - 1,
            rng,
        )
        return [ZERO_STATE, *states_strat]
    raise ValueError(f"Unsupported CAL_MODE: {CAL_MODE}")


def transform_matrix(state: MotionState) -> np.ndarray:
    rotation = Rotation.from_euler(
        "xyz",
        [state.x_rot_deg, state.y_rot_deg, state.z_rot_deg],
        degrees=True,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = np.array(
        [
            state.x_trans_mul * L,
            state.y_trans_mul * L,
            state.z_trans_mul * L,
        ],
        dtype=np.float64,
    )
    return transform


def apply_transform(coords: np.ndarray, transform: np.ndarray) -> np.ndarray:
    coords_h = np.hstack([coords, np.ones((coords.shape[0], 1), dtype=np.float64)])
    coords_transformed_h = (transform @ coords_h.T).T
    return coords_transformed_h[:, :3]


def displacement_fields(
    coords_ref: np.ndarray,
    states: list[MotionState],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes_num = coords_ref.shape[0]
    frames_num = len(states)
    disp_x = np.zeros((nodes_num, frames_num), dtype=np.float64)
    disp_y = np.zeros((nodes_num, frames_num), dtype=np.float64)
    disp_z = np.zeros((nodes_num, frames_num), dtype=np.float64)

    for ff, state in enumerate(states):
        transform = transform_matrix(state)
        coords_curr = apply_transform(coords_ref, transform)
        disp_curr = coords_curr - coords_ref
        disp_x[:, ff] = disp_curr[:, 0]
        disp_y[:, ff] = disp_curr[:, 1]
        disp_z[:, ff] = disp_curr[:, 2]

    return disp_x, disp_y, disp_z


def state_table(states: list[MotionState]) -> np.ndarray:
    rows = []
    for ff, state in enumerate(states):
        rows.append(
            [
                ff,
                state.x_trans_mul,
                state.y_trans_mul,
                state.z_trans_mul,
                state.x_trans_mul * L,
                state.y_trans_mul * L,
                state.z_trans_mul * L,
                state.x_rot_deg,
                state.y_rot_deg,
                state.z_rot_deg,
            ]
        )
    return np.array(rows, dtype=np.float64)


def save_csv_matrix(file_path: Path, data: np.ndarray, fmt: str) -> None:
    np.savetxt(file_path, data, delimiter=",", fmt=fmt)


def write_case(
    case_name: str,
    coords: np.ndarray,
    connect: np.ndarray,
    uvs: np.ndarray,
    states: list[MotionState],
) -> None:
    out_dir = BASE_DIR / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    disp_x, disp_y, disp_z = displacement_fields(coords, states)

    save_csv_matrix(out_dir / "coords.csv", coords, "%.10f")
    save_csv_matrix(out_dir / "connect.csv", connect, "%d")
    save_csv_matrix(out_dir / "uvs.csv", uvs, "%.10f")
    save_csv_matrix(out_dir / "field_disp_x.csv", disp_x, "%.10f")
    save_csv_matrix(out_dir / "field_disp_y.csv", disp_y, "%.10f")
    save_csv_matrix(out_dir / "field_disp_z.csv", disp_z, "%.10f")

    states_arr = state_table(states)
    states_header = (
        "frame_idx,x_trans_mul,y_trans_mul,z_trans_mul,"
        "x_trans,y_trans,z_trans,x_rot_deg,y_rot_deg,z_rot_deg"
    )
    np.savetxt(
        out_dir / "states.csv",
        states_arr,
        delimiter=",",
        fmt="%.10f",
        header=states_header,
        comments="",
    )


def main() -> None:
    states = selected_motion_states()
    cases = mesh_cases()

    for case_name, (coords, connect, uvs) in cases.items():
        write_case(case_name, coords, connect, uvs, states)

    print(f"Generated {len(cases)} calplate mesh cases in {BASE_DIR}")
    print(f"Mode: {CAL_MODE}")
    print(f"Frames per case: {len(states)}")
    print(f"Aspect ratio: {ASPECT_RATIO:.10f}")
    print(f"Length: {L:.10f} m")
    print(f"Short dimension: {SHORT_DIM:.10f} m")


if __name__ == "__main__":
    main()
