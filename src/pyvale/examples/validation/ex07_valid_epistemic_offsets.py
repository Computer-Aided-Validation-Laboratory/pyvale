# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Epistemic offsets
================================================================================

Propagate coherent experimental offsets and sample confidence through MAVM.
"""

from itertools import count
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.validation as vl

# %%
matplotlib.use("Agg")

SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex07_epistemic_offsets"
output_path.mkdir(parents=True, exist_ok=True)
fig_nums = count(1)

# %%
# 1. Build nominal samples and offsets
# ------------------------------------
#
# The sample arrays have point, component, time, and aleatory axes. At each
# point and time, the lower and upper offset magnitudes describe a coherent
# shift applied to every experimental repeat.
coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
times = np.array([0.0, 1.0, 2.0])
ground_truth = np.array([100.0, 120.0])[:, None] + 10.0 * times

aleatory_deviations = np.array([-6.0, -2.0, 2.0, 6.0])
exp_bias = 4.0 + 5.0 * times
sim_samples = ground_truth[:, None, :, None] + aleatory_deviations
exp_samples = (ground_truth + exp_bias)[:, None, :, None] + aleatory_deviations

exp_epistemic_offset_lower = np.array([1.5, 2.5])[:, None] + 0.5 * times
exp_epistemic_offset_upper = np.array([4.0, 6.0])[:, None] + times

# %%
# 2. Attach offsets to the experiment
# -----------------------------------
#
# ``EpistemicOffsets`` takes positive magnitudes for the downward and upward
# shifts. Its point and time axes let the magnitudes vary across the plate and
# over the measurement history.
axes = (
    vl.EValidAxis.POINT,
    vl.EValidAxis.COMPONENT,
    vl.EValidAxis.TIME,
    vl.EValidAxis.ALEATORY,
)

observable_tag = "temperature"

valid_case = vl.ValidCase(
    {
        observable_tag: vl.ValidObservable(
            components=("temperature",),
            geometry=vl.EValidGeometry.POINT,
            coords=coords,
            times=times,
            phys_units="degC",
            coord_units="m",
            time_units="s",
        )
    },
    {observable_tag: vl.ValidArray(sim_samples, axes)},
    {
        observable_tag: vl.ValidArray(
            exp_samples,
            axes,
            vl.EpistemicOffsets(
                exp_epistemic_offset_lower,
                exp_epistemic_offset_upper,
                (vl.EValidAxis.POINT, vl.EValidAxis.TIME),
            ),
        )
    },
)

# %%
# 3. Calculate MAVM for each offset
# ---------------------------------
#
# The metric result retains an experiment offset axis with nominal, lower, and
# upper selections. Use this axis to select the measurement assumption for a
# time plot.
mavm = vl.MetricMAVM()
valid_result = vl.ValidEngine([mavm]).run(valid_case)

# %%
# 4. Show the outer confidence limits
# -----------------------------------
#
# Select all experimental offsets with ``exp_offset=None``. The confidence
# level comes from the MAVM significance setting. The plot shows the nominal
# and offset cases with the outer confidence bounds, and the MAVM overlay adds
# the support span.
ecdf_plot = vl.plot_sim_exp_cdfs(
    valid_case,
    observable_tag,
    time_idx=2,
    point_idxs=None,
    exp_offset=None,
    confidence_alpha=mavm.alpha,
)
vl.add_metric_overlay(ecdf_plot, mavm)

fig_file = f"val_ex07_fig{next(fig_nums):02d}_mavm_offset_confidence_ecdfs.png"
ecdf_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

if not SHOW_PLOTS:
    plt.close(ecdf_plot.figure)

# %%
# .. image:: /_static/val_ex07_fig01_mavm_offset_confidence_ecdfs.png
#    :alt: Experimental offset ECDFs, outer confidence limits, and MAVM support
#    :width: 100%
#    :align: center

# %%
# 5. Plot the nominal metric history
# ----------------------------------
#
# Select the nominal experiment offset to track both MAVM directions for the
# two sites over time.
fig, _ = vl.plot_metric_time_points(
    valid_result,
    observable_tag,
    vl.EMetricType.MAVM,
    point_idxs=(0, 1),
    selectors={vl.EValidResAxis.EXP_OFFSET: vl.EEpisOffset.NOMINAL},
)
fig_file = f"val_ex07_fig{next(fig_nums):02d}_mavm_time.png"
fig.savefig(output_path / fig_file, bbox_inches="tight")
if not SHOW_PLOTS:
    plt.close(fig)

# %%
# .. image:: /_static/val_ex07_fig02_mavm_time.png
#    :alt: Directional MAVM histories for the nominal experiment
#    :width: 100%
#    :align: center

if SHOW_PLOTS:
    plt.show()
