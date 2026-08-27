# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from collections.abc import Callable
import numpy as np
from scipy.optimize import newton
from pyvale.sensorsim.errorsimulator import (
    IErrSimulator,
    EErrType,
    EErrDep,
)
from pyvale.sensorsim.sensordata import SensorData


class ErrSysCalibration(IErrSimulator):
    """Systematic error calculator for calibration errors.

    The user specifies an assumed calibration and a ground truth calibration
    function. Inversion can be performed via numerical lookup table with linear
    interpolation or exact Newton-Raphson root finding.

    Implements the `IErrSimulator` interface.
    """

    __slots__ = (
        "_assumed_calib",
        "_truth_calib",
        "_truth_calib_prime",
        "_cal_range",
        "_n_cal_divs",
        "_use_newton",
        "_tol",
        "_max_iter",
        "_err_dep",
        "_truth_cal_table",
    )

    def __init__(
        self,
        assumed_calib: Callable[[np.ndarray], np.ndarray],
        truth_calib: Callable[[np.ndarray], np.ndarray],
        cal_range: tuple[float, float],
        n_cal_divs: int = 10000,
        use_newton: bool = False,
        truth_calib_prime: Callable[[np.ndarray], np.ndarray] | None = None,
        tol: float = 1e-8,
        max_iter: int = 50,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
    ) -> None:
        """
        Parameters
        ----------
        assumed_calib : Callable[[np.ndarray], np.ndarray]
            Assumed calibration function converting raw signal to physical
            units.
        truth_calib : Callable[[np.ndarray], np.ndarray]
            Ground truth calibration function converting raw signal to physical
            units.
        cal_range : tuple[float, float]
            Range over which the calibration functions are valid.
        n_cal_divs : int, optional
            Number of divisions for lookup table discretisation (default 10000).
        use_newton : bool, optional
            Whether to use Newton-Raphson exact root-finding for inversion
            (default False).
        truth_calib_prime : Callable[[np.ndarray], np.ndarray] | None, optional
            Derivative of truth calibration function for Newton-Raphson.
        tol : float, optional
            Tolerance for Newton root-finding (default 1e-8).
        max_iter : int, optional
            Maximum iterations for Newton root-finding (default 50).
        err_dep : EErrDep, optional
            Error calculation dependence (default EErrDep.INDEPENDENT).
        """
        self._assumed_calib = assumed_calib
        self._truth_calib = truth_calib
        self._truth_calib_prime = truth_calib_prime
        self._cal_range = cal_range
        self._n_cal_divs = n_cal_divs
        self._use_newton = use_newton
        self._tol = tol
        self._max_iter = max_iter
        self._err_dep = err_dep

        self._truth_cal_table = np.zeros((n_cal_divs, 2), dtype=np.float64)
        self._truth_cal_table[:, 0] = np.linspace(
            cal_range[0], cal_range[1], n_cal_divs
        )
        self._truth_cal_table[:, 1] = self._truth_calib(
            self._truth_cal_table[:, 0]
        )

    def get_error_dep(self) -> EErrDep:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDep) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def reseed(self, seed: int | None = None) -> None:
        pass

    def sim_errs(
        self,
        err_basis: np.ndarray,
        sens_data: SensorData,
    ) -> tuple[np.ndarray, SensorData]:
        # Initial guess from lookup table interpolation
        signal_from_field = np.interp(
            err_basis,
            self._truth_cal_table[:, 1],
            self._truth_cal_table[:, 0],
        )

        if self._use_newton:
            signal_from_field = _invert_calibration_newton(
                truth_calib=self._truth_calib,
                y_vals=err_basis,
                x0_init=signal_from_field,
                fprime=self._truth_calib_prime,
                tol=self._tol,
                max_iter=self._max_iter,
            )

        field_from_assumed_calib = self._assumed_calib(signal_from_field)
        sys_errs = field_from_assumed_calib - err_basis
        return sys_errs, sens_data


def _invert_calibration_newton(
    truth_calib: Callable[[np.ndarray], np.ndarray],
    y_vals: np.ndarray,
    x0_init: np.ndarray,
    fprime: Callable[[np.ndarray], np.ndarray] | None = None,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> np.ndarray:
    """Inverts the truth calibration function via Newton-Raphson root
    finding.
    """
    flat_y = np.asarray(y_vals, dtype=np.float64).ravel()
    flat_x0 = np.asarray(x0_init, dtype=np.float64).ravel()
    flat_res = np.empty_like(flat_y)

    for idx, (y_target, x_guess) in enumerate(
        zip(flat_y, flat_x0, strict=False)
    ):

        def obj_func(x: float) -> float:
            return float(truth_calib(np.array([x]))[0] - y_target)

        prime_func = None
        if fprime is not None:

            def prime_func(x: float) -> float:
                return float(fprime(np.array([x]))[0])

        try:
            root = newton(
                obj_func,
                x0=float(x_guess),
                fprime=prime_func,
                tol=tol,
                maxiter=max_iter,
            )
            flat_res[idx] = root
        except Exception:
            flat_res[idx] = x_guess

    return flat_res.reshape(y_vals.shape)

