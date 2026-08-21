# %%
"""Riley stereo-calibration parity demo through :mod:`pyvale.render`."""

from riley.pydemos.demo_stereocal import main

from _riley_demo_tools import run_demo


# %% This needs the preceding DIC-UQ calibration files in the working directory.
run_demo(main)
