#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

"""
DEVELOPER VERIFICATION MODULE
--------------------------------------------------------------------------------
This module contains developer utility functions used for verification testing
of the point sensor simulation toolbox in pyvale.

Specifically, this module contains generic functions used across all types of
point sensors.
"""

import numpy as np
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.dataio as io
import pyvale.verif.pointsensconst as pointsensconst
from pyvale.verif.pointsensconst import GOLD_SEED

def joggle_meshfree_coords(
    coords: np.ndarray,
    scale_factor: float = 1e-10,
    seed: int = GOLD_SEED,
) -> np.ndarray:
    """
    Apply a deterministic perturbation to coordinate point clouds to
    break geometric degeneracies (e.g. co-circular/co-spherical points) in
    Delaunay triangulation across platforms and architectures.

    Preserves bounding box planes so that sensors placed on outer faces remain
    inside the convex hull without producing out-of-bounds NaNs.

    Parameters
    ----------
    coords : np.ndarray
        Array of nodal coordinates with shape (n_nodes, 3).
    scale_factor : float, optional
        Scale factor for coordinate perturbation relative to mesh bounding box
        extent, by default 1e-10.
    seed : int, optional
        RNG seed for deterministic jitter, by default GOLD_SEED.

    Returns
    -------
    np.ndarray
        Perturbed coordinates array with shape (n_nodes, 3).
    """
    rng = np.random.default_rng(seed)
    char_length = float(np.ptp(coords, axis=0).max())
    jitter = scale_factor * char_length * rng.standard_normal(coords.shape)

    coord_min = coords.min(axis=0)
    coord_max = coords.max(axis=0)
    tol = 1e-7

    # Preserve bounding box faces to keep boundary sensors inside convex hull
    for dd in range(coords.shape[1]):
        mask_min = np.isclose(coords[:, dd], coord_min[dd], atol=tol)
        mask_max = np.isclose(coords[:, dd], coord_max[dd], atol=tol)
        jitter[mask_min, dd] = 0.0
        jitter[mask_max, dd] = 0.0

    if np.allclose(coords[:, 2], 0.0):
        jitter[:, 2] = 0.0

    return coords + jitter


def samp_times(sim_data: io.SimData) -> dict[str, None | np.ndarray]:
    sim_dims = sens.simtools.get_sim_dims(sim_data)
    sample_times = {}

    sample_times["sim"] = None
    sample_times["user"] = np.linspace(0.0,sim_dims["t"][1],50)

    return sample_times


def sens_data_dict(sim_data: io.SimData,
                   sens_pos: dict[str,np.ndarray]) -> dict[str,sens.SensorData]:
    sample_times = samp_times(sim_data)

    sens_data = {}
    for pp in sens_pos:
        for tt in sample_times:
            tag = f"pos-{pp}_time-{tt}"
            sens_data[tag] = sens.SensorData(
                positions=sens_pos[pp],
                sample_times=sample_times[tt],
            )

    return sens_data


def err_chain_basic() -> list[sens.IErrSimulator]:
    chain_basic = [
        sens.ErrSysOffset(offset=-1.0),
        sens.ErrSysGen(sens.GenUniform(low=-1.0,high=1.0,seed=GOLD_SEED)),
        sens.ErrSysGenPercent(
            sens.GenUniform(low=-1.0,high=1.0,seed=GOLD_SEED)),
        sens.ErrRandGen(sens.GenNormal(std=1.0,seed=GOLD_SEED)),
        sens.ErrRandGenPercent(sens.GenNormal(std=1.0,seed=GOLD_SEED)),
    ]
    return chain_basic


def err_chain_gen() -> list[sens.IErrSimulator]:
    chain_gen = [
        sens.ErrSysOffset(offset=-1.0),
        sens.ErrSysGen(sens.GenUniform(low=-1.0,high=1.0,seed=GOLD_SEED)),
        sens.ErrSysGenPercent(
            sens.GenUniform(low=-1.0,high=1.0,seed=GOLD_SEED)),
        sens.ErrRandGen(sens.GenNormal(std=1.0,seed=GOLD_SEED)),
        sens.ErrRandGenPercent(sens.GenNormal(std=1.0,seed=GOLD_SEED)),
    ]
    return chain_gen


def err_chain_dep() -> list[sens.IErrSimulator]:
    chain_dep = [
        sens.ErrSysRoundOff(sens.ERoundMethod.ROUND,0.1),
        sens.ErrSysDigitisation(bits_per_unit=2**16/100),
        sens.ErrSysSaturation(meas_min=0.0,meas_max=100.0),
    ]
    return chain_dep


def err_chain_all(err_dict: dict[str,list[sens.IErrSimulator]]
                  ) -> list[sens.IErrSimulator]:
    err_chain = []
    for ee in err_dict:
        if err_dict[ee] is not None:
            for ss in err_dict[ee]:
                err_chain.append(ss)
    return err_chain


def gen_gold_measurements(sens_dict: dict[str,sens.SensorsPoint]) -> None:
    for ss in sens_dict:
        print(f"Generating gold output for case: {ss}")
        measurements = sens_dict[ss].sim_measurements()
        save_path = pointsensconst.GOLD_PATH / f"{ss.lower()}.npy"
        np.save(save_path,measurements)


def check_gold_measurements(sens_dict: dict[str,sens.SensorsPoint],
                            rtol: float = 1e-5,
                            atol: float = 1e-5) -> list[str]:
    fails = []

    for ss in sens_dict:
        measurements = sens_dict[ss].sim_measurements()
        gold_path = pointsensconst.GOLD_PATH / f"{ss.lower()}.npy"

        if gold_path.is_file():
            gold = np.load(gold_path)

            if not np.allclose(measurements,gold,rtol=rtol,atol=atol):
                fails.append(f"Gold check failed for: {ss}")
        else:
            fails.append(
                f"Gold file does not exist for: {ss}, path: {gold_path}"
            )

    return fails


