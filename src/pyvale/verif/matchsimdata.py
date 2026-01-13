#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import enum
import numpy as np
from pyvale.mooseherder.simdata import SimData


class EDataCheck(enum.Enum):
    PASS_BOTH_NONE = enum.auto()
    FAIL_DATA0_NONE = enum.auto()
    FAIL_DATA1_NONE = enum.auto()
    FAIL_DICT_KEY_MISMATCH = enum.auto()
    FAIL_SHAPE_MISMATCH = enum.auto()
    FAIL_DATA_MISMATCH = enum.auto()
    PASS_DATA_MATCH = enum.auto()

    def is_pass(self) -> bool:
        return self.name.startswith("PASS")


def match_sim_data(data0: SimData, data1: SimData) -> dict[str,EDataCheck]:
    data_checks = {}

    data_checks["coords"] = _check_sim_data_field(data0.coords,data1.coords)
    data_checks["time"] = _check_sim_data_field(data0.time,data1.time)
    data_checks["connect"] = _check_sim_data_dict(data0.connect,data1.connect)
    data_checks["glob_vars"] = _check_sim_data_dict(data0.glob_vars,
                                                    data1.glob_vars)
    data_checks["node_vars"] = _check_sim_data_dict(data0.node_vars,
                                                    data1.node_vars)
    data_checks["elem_vars"] = _check_sim_data_dict(data0.elem_vars,
                                                    data1.elem_vars)
    return data_checks


def match_sim_data_get_fails(data0: SimData, data1: SimData) -> list[str]:
    data_checks = match_sim_data(data0,data1)

    fails = []
    for dd in data_checks:
        if isinstance(data_checks[dd],dict):
            if data_checks[dd] is not None:
                for kk in data_checks[dd]:
                    if not data_checks[dd][kk].is_pass():
                        fails.append(f"{dd}, {kk}: {data_checks[dd][kk].name}")
        else:
            if data_checks[dd] is not None:
                if not data_checks[dd].is_pass():
                    fails.append(f"{dd}: {data_checks[dd].name}")

    return fails





def _check_sim_data_field(data0: np.ndarray | None,
                          data1: np.ndarray | None
                          ) -> EDataCheck:

    if data0 is None and data1 is None:
        return EDataCheck.PASS_BOTH_NONE

    if data0 is None and data1 is not None:
        return EDataCheck.FAIL_DATA0_NONE

    if data0 is not None and data1 is None:
        return EDataCheck.FAIL_DATA1_NONE

    if data0.shape != data1.shape:
        return EDataCheck.FAIL_SHAPE_MISMATCH

    if not np.allclose(data0,data1):
        return EDataCheck.FAIL_DATA_MISMATCH

    return EDataCheck.PASS_DATA_MATCH


def _check_sim_data_dict(data0: dict[str,np.ndarray] | None,
                         data1: dict[str,np.ndarray] | None,
                         ) -> dict[str,EDataCheck] | EDataCheck:

    if data0 is None and data1 is None:
        return EDataCheck.PASS_BOTH_NONE

    if data0 is None and data1 is not None:
        return EDataCheck.FAIL_DATA0_NONE

    if data0 is not None and data1 is None:
        return EDataCheck.FAIL_DATA1_NONE

    if data0.keys() != data1.keys():
        return EDataCheck.FAIL_DICT_KEY_MISMATCH

    data_check = {}
    for kk in data0:
        if data0[kk].shape != data1[kk].shape:
            data_check[kk] = EDataCheck.FAIL_SHAPE_MISMATCH
            continue

        if not np.allclose(data0[kk],data1[kk]):
            data_check[kk] = EDataCheck.FAIL_DATA_MISMATCH
            continue

        data_check[kk] = EDataCheck.PASS_DATA_MATCH

    return data_check


