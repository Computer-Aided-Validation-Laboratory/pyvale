#!/usr/bin/env python3
from __future__ import annotations

import pathlib


SHEAR_REGULAR = 10
SHEAR_SHEAR = 0

TRI6_BULGE_OUT_LIMIT = 0
TRI6_BULGE_OUT = 1
TRI6_BULGE_REGULAR = 6
TRI6_BULGE_IN = 8
TRI6_BULGE_IN_LIMIT = 9

QUAD89_BULGE_OUT_LIMIT = 1
QUAD89_BULGE_OUT = 2
QUAD89_BULGE_REGULAR = 6
QUAD89_BULGE_IN = 9
QUAD89_BULGE_IN_LIMIT = 10

TABLE_MEDIAN_DECIMAL_PLACES = 2
TABLE_MAD_DECIMAL_PLACES = 3

PLOT_RESOLUTION_DPI = 300.0
PLOT_SQUARE_FIG_SIZE_IN = (3.625, 3.625)
PLOT_LINE_FIG_SIZE_IN = (3.625, 3.10)
PLOT_AXIS_FONT_SIZE = 24.0
PLOT_TICK_FONT_SIZE = 23.0
PLOT_TITLE_FONT_SIZE = 24.0
PLOT_LEGEND_FONT_SIZE = 23.0
PLOT_LINE_AXIS_FONT_SIZE = 10.0
PLOT_LINE_TICK_FONT_SIZE = 10.0
PLOT_LINE_TITLE_FONT_SIZE = 14.0
PLOT_LINE_LEGEND_FONT_SIZE = 10.0
PLOT_LINE_SECONDARY_AXIS_FONT_SIZE = 10.0

PAPER_DIR = pathlib.Path("~/ray_tracing_verification/verif_1").expanduser()


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def is_quad89(mesh_name: str) -> bool:
    return mesh_name in {"quad8", "quad9"}


def bulge_out_limit_frame(mesh_name: str) -> int:
    if is_quad89(mesh_name):
        return QUAD89_BULGE_OUT_LIMIT
    return TRI6_BULGE_OUT_LIMIT


def bulge_out_frame(mesh_name: str) -> int:
    if is_quad89(mesh_name):
        return QUAD89_BULGE_OUT
    return TRI6_BULGE_OUT


def bulge_regular_frame(mesh_name: str) -> int:
    if is_quad89(mesh_name):
        return QUAD89_BULGE_REGULAR
    return TRI6_BULGE_REGULAR


def bulge_in_frame(mesh_name: str) -> int:
    if is_quad89(mesh_name):
        return QUAD89_BULGE_IN
    return TRI6_BULGE_IN


def bulge_in_limit_frame(mesh_name: str) -> int:
    if is_quad89(mesh_name):
        return QUAD89_BULGE_IN_LIMIT
    return TRI6_BULGE_IN_LIMIT
