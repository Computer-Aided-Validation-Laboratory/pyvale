# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import pytest
import numpy as np
from scipy.spatial.transform import Rotation
from pyvale.sensorsim.fieldtransforms import (
    FieldTransformCustom,
    FieldTransformVonMises,
    FieldTransformPrincipal,
    FieldTransformHydrostatic,
    FieldTransformTraction,
    FieldTransformFlux,
    FieldTransformMagnitude,
    FieldTransformDirectional,
    FieldTransformChain,
)


def test_von_mises_2d() -> None:
    vm_trans = FieldTransformVonMises()
    # Case 1: Uniaxial tension sigma_xx = 100.0, sigma_yy = 0, sigma_xy = 0
    raw_1 = np.array([[[100.0]], [[0.0]], [[0.0]]]).transpose(1, 0, 2)
    pts = np.zeros((1, 3))
    times = np.array([0.0])
    res_1 = vm_trans.transform(raw_1, pts, times)
    assert np.isclose(res_1[0, 0, 0], 100.0)

    # Case 2: Pure shear sigma_xy = 50.0 -> von Mises = sqrt(3)*50 = 86.6025
    raw_2 = np.array([[[0.0]], [[0.0]], [[50.0]]]).transpose(1, 0, 2)
    res_2 = vm_trans.transform(raw_2, pts, times)
    assert np.isclose(res_2[0, 0, 0], np.sqrt(3.0) * 50.0)


def test_von_mises_3d() -> None:
    vm_trans = FieldTransformVonMises()
    # Case 1: Hydrostatic stress -> von Mises = 0
    raw_hydro = np.zeros((1, 6, 1))
    raw_hydro[0, 0, 0] = -200.0  # s_xx
    raw_hydro[0, 1, 0] = -200.0  # s_yy
    raw_hydro[0, 2, 0] = -200.0  # s_zz
    pts = np.zeros((1, 3))
    times = np.array([0.0])
    res_hydro = vm_trans.transform(raw_hydro, pts, times)
    assert np.isclose(res_hydro[0, 0, 0], 0.0)

    # Case 2: 3D pure shear on all planes -> s_xy=s_yz=s_xz=10
    raw_shear = np.zeros((1, 6, 1))
    raw_shear[0, 3, 0] = 10.0
    raw_shear[0, 4, 0] = 10.0
    raw_shear[0, 5, 0] = 10.0
    res_shear = vm_trans.transform(raw_shear, pts, times)
    # vm_sq = 0.5 * 6 * 3 * 100 = 900 -> vm = 30
    assert np.isclose(res_shear[0, 0, 0], 30.0)


def test_principal_2d() -> None:
    p_trans = FieldTransformPrincipal(return_max_shear=True)
    # Pure shear: s_xx=0, s_yy=0, s_xy=40 -> p1 = 40, p2 = -40, gamma_max = 80
    raw = np.array([[[0.0]], [[0.0]], [[40.0]]]).transpose(1, 0, 2)
    pts = np.zeros((1, 3))
    times = np.array([0.0])
    res = p_trans.transform(raw, pts, times)
    assert np.isclose(res[0, 0, 0], 40.0)
    assert np.isclose(res[0, 1, 0], -40.0)
    assert np.isclose(res[0, 2, 0], 80.0)


def test_principal_3d() -> None:
    p_trans = FieldTransformPrincipal(return_max_shear=True)
    # Diagonal tensor diag(100, 50, -20)
    raw = np.zeros((1, 6, 1))
    raw[0, 0, 0] = 100.0
    raw[0, 1, 0] = 50.0
    raw[0, 2, 0] = -20.0
    pts = np.zeros((1, 3))
    times = np.array([0.0])
    res = p_trans.transform(raw, pts, times)
    assert np.isclose(res[0, 0, 0], 100.0)
    assert np.isclose(res[0, 1, 0], 50.0)
    assert np.isclose(res[0, 2, 0], -20.0)
    # tau_max = (100 - (-20)) / 2 = 60
    assert np.isclose(res[0, 3, 0], 60.0)


def test_traction_and_flux() -> None:
    tr_trans = FieldTransformTraction(include_scalar_projections=True)
    # Tensor diag(10, 20, 30), normal is z-axis (default)
    raw_tensor = np.zeros((1, 6, 1))
    raw_tensor[0, 0, 0] = 10.0
    raw_tensor[0, 1, 0] = 20.0
    raw_tensor[0, 2, 0] = 30.0
    pts = np.zeros((1, 3))
    times = np.array([0.0])
    res_tr = tr_trans.transform(raw_tensor, pts, times)
    # tx=0, ty=0, tz=30, t_n=30, t_s=0
    assert np.isclose(res_tr[0, 0, 0], 0.0)
    assert np.isclose(res_tr[0, 1, 0], 0.0)
    assert np.isclose(res_tr[0, 2, 0], 30.0)
    assert np.isclose(res_tr[0, 3, 0], 30.0)
    assert np.isclose(res_tr[0, 4, 0], 0.0)

    # Flux test: q = (10, 20, 30), normal = (0, 0, 1) -> q_n = 30
    fl_trans = FieldTransformFlux()
    raw_vec = np.array([[[10.0]], [[20.0]], [[30.0]]]).transpose(1, 0, 2)
    res_fl = fl_trans.transform(raw_vec, pts, times)
    assert np.isclose(res_fl[0, 0, 0], 30.0)


def test_custom_and_chain() -> None:
    # Custom failure index: (x / 100)^2 + (y / 50)^2
    custom_func = lambda raw, p, t, a: (raw[:, 0:1, :] / 100.0) ** 2 + (
        raw[:, 1:2, :] / 50.0
    ) ** 2
    c_trans = FieldTransformCustom(custom_func, component_names=("fail_idx",))
    raw = np.array([[[100.0]], [[50.0]]]).transpose(1, 0, 2)
    pts = np.zeros((1, 3))
    times = np.array([0.0])
    res = c_trans.transform(raw, pts, times)
    assert np.isclose(res[0, 0, 0], 2.0)

    # Magnitude
    mag_trans = FieldTransformMagnitude()
    raw_vec = np.array([[[3.0]], [[4.0]]]).transpose(1, 0, 2)
    res_mag = mag_trans.transform(raw_vec, pts, times)
    assert np.isclose(res_mag[0, 0, 0], 5.0)

    # Chain: magnitude then scaling by 2.0
    scale_trans = FieldTransformCustom(
        lambda raw, p, t, a: raw * 2.0, component_names=("scaled_mag",)
    )
    chain = FieldTransformChain([mag_trans, scale_trans])
    res_chain = chain.transform(raw_vec, pts, times)
    assert np.isclose(res_chain[0, 0, 0], 10.0)
