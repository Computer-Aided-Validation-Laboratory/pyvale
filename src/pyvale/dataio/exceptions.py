# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

class SimLoadErr(Exception):
    """Custom exception for errors when loading simulation data from file.
    """
    pass


class ExpLoadErr(Exception):
    """Custom exception for errors when loading experimental data from file
    """
    pass
