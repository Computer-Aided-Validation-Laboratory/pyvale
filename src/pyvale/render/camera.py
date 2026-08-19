# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Camera data for three-dimensional and planar rendering."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(slots=True)
class Camera:
    """A perspective camera that maps directly to Riley camera capabilities."""

    pixels_num: np.ndarray
    pixels_size: np.ndarray
    pos_world: np.ndarray
    rot_world: Rotation
    roi_cent_world: np.ndarray
    focal_length: float
    sub_sample: int = 1
    distortion_model: int = 0
    distortion_k1: float = 0.0
    distortion_k2: float = 0.0
    distortion_k3: float = 0.0
    distortion_k4: float = 0.0
    distortion_k5: float = 0.0
    distortion_k6: float = 0.0
    distortion_p1: float = 0.0
    distortion_p2: float = 0.0
    c0: float | None = None
    c1: float | None = None
    fstop: float | None = None
    psf_type: int = 0
    psf_sigma_x: float = 0.0
    psf_sigma_y: float = 0.0
    psf_theta: float = 0.0
    psf_support_rad: float = 0.0

    def __post_init__(self) -> None:
        self.pixels_num = np.asarray(self.pixels_num, dtype=np.int32)
        self.pixels_size = np.asarray(self.pixels_size, dtype=np.float64)
        self.pos_world = np.asarray(self.pos_world, dtype=np.float64)
        self.roi_cent_world = np.asarray(self.roi_cent_world, dtype=np.float64)
        if self.c0 is None:
            self.c0 = float(self.pixels_num[0]) / 2.0
        if self.c1 is None:
            self.c1 = float(self.pixels_num[1]) / 2.0


@dataclass(slots=True)
class Camera2D:
    """An orthographic camera for planar image-warp renderers."""

    pixels_count: np.ndarray = field(
        default_factory=lambda: np.array((1000, 1000), dtype=np.int32),
    )
    leng_per_px: float = 1.0e-3
    bits: int = 8
    roi_cent_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64),
    )
    background: float = 0.5
    sample_times: np.ndarray | None = None
    angle: Rotation | None = None
    subsample: int = 2
    field_of_view: np.ndarray = field(init=False)
    dynamic_range: int = field(init=False)
    world_to_cam: np.ndarray = field(init=False)
    cam_to_world: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.pixels_count = np.asarray(self.pixels_count, dtype=np.int32)
        self.roi_cent_world = np.asarray(self.roi_cent_world, dtype=np.float64)
        self.field_of_view = self.leng_per_px * self.pixels_count.astype(np.float64)
        self.dynamic_range = 2 ** self.bits
        self.background *= float(self.dynamic_range)
        self.world_to_cam = self.field_of_view / 2.0 - self.roi_cent_world[:2]
        self.cam_to_world = -self.world_to_cam


__all__ = ["Camera", "Camera2D"]
