"""
Riley Exodus DIC-UQ parity demo through :mod:`pyvale.render`.
=============================================================
"""


from riley.pydemos.demo_dic_from_exodus import main

from _riley_demo_tools import run_demo


# %% The packaged Riley scene is rendered through render.Riley and Scene3D.
run_demo(main)
