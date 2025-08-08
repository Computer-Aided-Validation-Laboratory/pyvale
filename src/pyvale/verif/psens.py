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
import pyvale.verif.psensconst as psensconst


def samp_times(sim_data: mh.SimData) -> dict[str, None | np.ndarray]:
    sim_dims = sens.SimTools.get_sim_dims(sim_data)
    sample_times = {}

    sample_times["sim"] = None
    sample_times["user"] = np.linspace(0.0,sim_dims["t"][1],50)

    return sample_times


def sens_data_dict(sim_data: mh.SimData,
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


def err_chain_basic() -> list[sens.IErrCalculator]:
    chain_basic = []
    chain_basic.append(sens.ErrSysOffset(offset=-1.0))
    chain_basic.append(sens.ErrSysUnif(low=-1.0,
                                      high=1.0,
                                      seed=psensconst.GOLD_SEED))
    chain_basic.append(sens.ErrSysUnifPercent(low_percent=-1.0,
                                             high_percent=1.0,
                                             seed=psensconst.GOLD_SEED))
    chain_basic.append(sens.ErrRandNorm(std=1.0,
                                       seed=psensconst.GOLD_SEED))
    chain_basic.append(sens.ErrRandNormPercent(std_percent=1.0,
                                              seed=psensconst.GOLD_SEED))
    return chain_basic


def err_chain_gen() -> list[sens.IErrCalculator]:
    chain_gen = []
    chain_gen.append(sens.ErrSysOffset(offset=-1.0))
    chain_gen.append(sens.ErrSysGen(
        sens.GenUniform(low=-1.0,high=1.0,seed=psensconst.GOLD_SEED)))
    chain_gen.append(sens.ErrSysGenPercent(
        sens.GenUniform(low=-1.0,high=1.0,seed=psensconst.GOLD_SEED)))
    chain_gen.append(sens.ErrRandGen(
        sens.GenNormal(std=1.0,seed=psensconst.GOLD_SEED)))
    chain_gen.append(sens.ErrRandGenPercent(
        sens.GenNormal(std=1.0,seed=psensconst.GOLD_SEED)))
    return chain_gen


def err_chain_dep() -> list[sens.IErrCalculator]:
    chain_dep = []
    chain_dep.append(sens.ErrSysRoundOff(sens.ERoundMethod.ROUND,0.1))
    chain_dep.append(sens.ErrSysDigitisation(bits_per_unit=2**16/100))
    chain_dep.append(sens.ErrSysSaturation(meas_min=0.0,meas_max=100.0))
    return chain_dep


def err_chain_all(err_dict: dict[str,list[sens.IErrCalculator]]
                  ) -> list[sens.IErrCalculator]:
    err_chain = []
    for ee in err_dict:
        if err_dict[ee] is not None:
            for ss in err_dict[ee]:
                err_chain.append(ss)
    return err_chain


def gen_gold_measurements(sens_dict: dict[str,sens.SensorArrayPoint]) -> None:
    for ss in sens_dict:
        print(f"Generating gold output for case: {ss}")
        measurements = sens_dict[ss].calc_measurements()
        save_path = psensconst.GOLD_PATH / f"{ss}.npy"
        np.save(save_path,measurements)


def check_gold_measurements(sens_dict: dict[str,sens.SensorArrayPoint]) -> list[str]:
    fails = []

    for ss in sens_dict:
        measurements = sens_dict[ss].calc_measurements()
        gold_path = psensconst.GOLD_PATH / f"{ss}.npy"

        load_path = psensconst.GOLD_PATH / f"{ss.lower()}.npy"
        if load_path.is_file():
            gold = np.load(load_path)

            if not np.allclose(measurements,gold):
                fails.append(f"Gold check failed for: {ss}")
        else:
            fails.append(f"Gold file does not exist for: {ss}, path: {gold_path}")

    return fails

def gen_gold_experiments(exp_sims: dict[str,sens.ExperimentSimulator]) -> None:
    pass

