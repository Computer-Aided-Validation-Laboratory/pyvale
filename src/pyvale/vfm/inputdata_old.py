from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class InputData:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    strain: npt.NDArray[np.float64]
    force: npt.NDArray[np.float64]
    time: npt.NDArray[np.float64]

    @classmethod
    def load_from_dir(cls, dir: Path) -> Self:
        # TODO: confirm file names
        x = _load_file("x_coordinates", dir)
        y = _load_file("y_coordinates", dir)

        strain = _load_file("strain", dir, allowed_extensions={".npy"})
        force = _load_force(dir)
        time = _load_timesteps(dir)

        input_data =  cls(
            x,
            y,
            strain,
            force,
            time,
        )

        return input_data


def _load_force(
    dir: Path,
    allowed_extensions: set[str] = {".txt", ".csv", ".npy"}
) -> npt.NDArray[np.float64]:
    # TODO: confirm file name
    content = _load_file("reaction_history", dir, allowed_extensions)

    if content.dtype.names is None:
        return content

    numeric_cols = [
        n for n in content.dtype.names
        if np.issubdtype(content.dtype[n], np.number)
    ]

    if not numeric_cols:
        raise ValueError("Force file contained no columns with numeric data")

    numeric_content = content[numeric_cols]

    possible_col_names = {"force", "load"}

    default_index = 0

    for name in numeric_content.dtype.names:
        if name in possible_col_names:
            default_index = content.dtype.names.index(name)

    _print_array_preview(numeric_content)

    index = _prompt_for_column_index(
        "force",
        [i for i in range(len(numeric_content.dtype.names))],
        default_index
    )

    col = numeric_content.dtype.names[index]

    unit = _prompt_for_unit("force", valid_units={"N", "kN"})

    data = numeric_content[col]

    if unit == "kN":
        data *= 1000

    if _prompt_should_flip_force_sign():
        data *= -1

    return data


def _load_timesteps(
    dir: Path,
    allowed_extensions: set[str] = {".txt", ".csv", ".npy"}
) -> npt.NDArray[np.float64]:
    # TODO: confirm file name
    content = _load_file("time_values", dir, allowed_extensions)

    if content.dtype.names is None:
        return content

    numeric_cols = [
        n for n in content.dtype.names
        if np.issubdtype(content.dtype[n], np.number)
    ]

    if not numeric_cols:
        raise ValueError(
            "Timesteps file contained no columns with numeric data"
        )

    numeric_content = content[numeric_cols]

    possible_col_names = {"time", "timestamp", "timesteps"}

    default_index = 0

    for name in numeric_content.dtype.names:
        if name in possible_col_names:
            default_index = content.dtype.names.index(name)

    _print_array_preview(numeric_content)

    index = _prompt_for_column_index(
        "force",
        [i for i in range(len(numeric_content.dtype.names))],
        default_index
    )

    col = numeric_content.dtype.names[index]

    data = numeric_content[col]

    if data[0] != 0.0:
        if _prompt_should_zero_time_offset(data[0]):
            data -= data[0]

    return data


def _load_file(
    file_name: str,
    dir: Path,
    allowed_extensions: set[str] = {".txt", ".csv", ".npy"}
) -> np.ndarray:
    file = next(dir.glob(f"{file_name}.*"), None)

    if file is None:
        raise FileNotFoundError(f"File '{file_name}' was not found")

    if file.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"{file.name} has an unsupported extension '{file.suffix}', "
            f"expected one of {allowed_extensions}"
        )

    if file.suffix.lower() == ".npy":
        contents = np.asarray(np.load(file), dtype=np.float64)
    elif file.suffix.lower() == ".csv":
        contents = np.genfromtxt(file, delimiter=",", names=True, dtype=None)
    else:
        contents = np.genfromtxt(file, delimiter=",", dtype=None)

    return contents


def _print_array_preview(arr: np.ndarray) -> None:
    if not arr.dtype.names:
        return

    num_rows = 5
    num_cols = len(arr.dtype.names)

    num_cols_per_batch = 5
    max_cell_width = 24


    preview_rows = np.atleast_1d(arr)[:num_rows]

    for batch_start in range(0, num_cols, num_cols_per_batch):
        batch_end = min(num_cols, batch_start + num_cols_per_batch)
        print(f"\nColumns {batch_start}-{batch_end - 1}:")

        header_cells = [
            f"[{index}] {arr.dtype.names[index]}"
            for index in range(batch_start, batch_end)
        ]

        row_cells = [
            [
                str(row[index])
                for index in range(batch_start, batch_end)
            ]
            for row in preview_rows
        ]

        column_widths = []
        for offset in range(batch_end - batch_start):
            cells = [
                header_cells[offset], *(row[offset] for row in row_cells)
            ]
            width = max(
                len(_truncate(cell, max_cell_width)) for cell in cells
            )
            column_widths.append(width)

        print(
            _format_preview_line(
                header_cells,
                column_widths,
                max_cell_width
            )
        )

        for row in row_cells:
            print(
                _format_preview_line(
                    row,
                    column_widths,
                    max_cell_width
                )
            )


def _format_preview_line(
    cells: list[str],
    column_widths: list[int],
    max_cell_width: int,
) -> str:
    formatted_cells = [
        _truncate(cell, max_cell_width).ljust(column_widths[index])
        for index, cell in enumerate(cells)
    ]
    return " | ".join(formatted_cells)


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else f"{text[: width - 3]}..."


def _prompt_for_column_index(
    name: str,
    indices: list[int],
    default_index: int,
) -> int:
    prompt = (
        f"Select the column index to use for {name}, "
        f"(available indices: {indices}, "
        f"default: {default_index}): "
    )

    while True:
        response = input(prompt).strip()

        if not response:
            return default_index

        try:
            selected = int(response)
        except ValueError:
            print("Please enter an integer column index")
            continue

        if selected not in indices:
            print(f"Column `{selected}` is invalid")
            continue

        return selected


def _prompt_for_unit(
    name: str,
    valid_units: set[str]
) -> str:
    while True:
        response = input(
            f"Enter the unit for {name}, "
            f"valid units are {valid_units}:"
        ).strip()

        if response in valid_units:
            return response
        else:
            print(f"{response} is not a valid unit, try again")
            continue


def _prompt_should_flip_force_sign() -> bool:
    response = input(
        "Multiply the force by -1? [y/N]: ").strip().lower()

    if response == "y":
        return True
    else:
        return False


def _prompt_should_zero_time_offset(initial_timestep: float) -> bool:
    response = input(
        f"Initial timestep is {initial_timestep}, should we "
        "subtract this offset so the first timestep becomes 0? [Y/n]: "
    ).strip().lower()

    if response == "n":
        return False
    else:
        return True
