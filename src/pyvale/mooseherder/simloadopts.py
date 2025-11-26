# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from dataclasses import dataclass


@dataclass(slots=True)
class SimLoadOpts:
    """Dataclass of options for loading simulation data from plain delimited
    text files.
    """

    delimiter: str = ","
    """Delimiter used to separate values in the plain text files to read.
    """

    coord_header: int | None = 0
    """Row indices to skip reading when loading plain text for the coordinate
    data file. Defaults to 0 which skips the first row as a header.
    """

    time_header: int | None = 0
    """Row indices to skip reading when loading plain text for the time
    data file. Defaults to 0 which skips the first row as a header.
    """

    connect_header: int | None = None
    """Row indices to skip reading when loading plain text for the connectivity
    data files. Defaults to 0 which skips the first row as a header.
    """

    glob_header: int | None = 0
    """Row indices to skip reading when loading plain text for the global
    variable data file. Defaults to 0 which skips the first row as a header.
    """

    node_field_header: int | None = 0
    """Row indices to skip reading when loading plain text for the nodal
    variables data files. Defaults to 0 which skips the first row as a header.
    """

    elem_field_header: int | None = 0
    """Row indices to skip reading when loading plain text for the element
    variables data files. Defaults to 0 which skips the first row as a header.
    """

    workers: int | None = None
    """Number of threads (i.e. multi-processing processes) to use when reading
    data files. Useful for reading many large data files in parallel. Defaults
    to None which is single threaded.
    """

