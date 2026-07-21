import numpy as np
import numpy.typing as npt

from pyvale.vfm.inputdata import InputDataConfig


def preprocess_input_data(config: InputDataConfig) -> None:
    x = config.x.load_from_file()
    y = config.y.load_from_file()
    strain = config.strain.load_from_file()
    force = config.force.load_from_file()
    time = config.time.load_from_file()
    roi = config.region_of_interest.load_from_file()

    _validate_input_data(
        x,
        y,
        strain,
        force,
        time
    )


    # Is data on a regular grid? if not need to interpolate

    # Enforce conventions
    # Validate data
    # generate plot
    # save plots
    # save results in a dir


def _validate_input_data(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    strain: npt.NDArray[np.float64],
    force: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64]
):
    errors: list[str] = []

    # Check dims
    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y)

    if force.ndim != 2:
        errors.append(
            f"Force must be a 2D array. Got a {force.ndim}D array."
        )

    if time.ndim != 1:
        errors.append(
            f"Time must be a 1D array. Got a {time.ndim}D array."
        )

    if strain.ndim != 4:
        errors.append(
            f"Strain must be a 4D array. Got a {strain.ndim}D array."
        )

    # Check shapes
    if x.shape != strain.shape[2:]:
        errors.append(
            f"Coordinate grid shape {x.shape} does not match spatial strain "
            f"components. Got shape {strain.shape[2:]}"
        )

    if force.shape[0] != time.shape[0]:
        errors.append(
            f"Number of rows in force ({force.shape[0]}) does not match the "
            f"number of timesteps ({time.shape[0]})."
        )

    if time.shape[0] != strain.shape[0]:
        errors.append(
            f"Number of timesteps ({time.shape[0]}) does not match the length "
            f"of the strain 0th dimension ({strain.shape[0]})."
        )

    if errors:
        raise ValueError(
            "Invalid input data:\n" + "\n".join(f"  - {e}" for e in errors)
        )
