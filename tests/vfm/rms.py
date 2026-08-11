import numpy as np
import numpy.typing as npt


def rms(array: npt.NDArray[np.float64]) -> float:
    return float(np.sqrt(np.nanmean(np.square(array))))


def root_mean_square_percentage_error(
    predicted: npt.NDArray[np.float64],
    known: npt.NDArray[np.float64],
) -> float:
    percentage_error = (predicted - known) / known * 100.0
    return float(np.sqrt(np.nanmean(np.square(percentage_error))))
