# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Validation metrics directly from NumPy arrays
================================================================================

Calculate metrics for one pair of one-dimensional NumPy samples. These
functions do not parallelise comparisons or interpret and reduce epistemic axes.
Use ``ValidEngine`` for those workflows, starting with
:doc:`Example 01a <ex01a_valid_scalar_metrics>`.
"""

import numpy as np

import pyvale.validation as vl

# %%
# 1. Create observations
# ----------------------
#
# The experiment is shifted by two units, as in Example 01a.
sim = np.array([0.0, 1.0, 2.0])
exp = sim + 2.0

# %%
# 2. Compare the distributions
# ----------------------------
#
# AVM, MAVM, Kolmogorov–Smirnov and Cramér–von Mises compare empirical
# distributions. The two samples can have different lengths. MAVM returns
# directional areas and the ECDF data used to calculate them. Use
# ``EMAVMDupes.INCLUDE`` when your observations can contain repeats.
avm = vl.calc_avm_1d(sim, exp)
mavm = vl.calc_mavm_1d(sim, exp)
ks = vl.calc_ks_1d(sim, exp)
cvm = vl.calc_cvm_1d(sim, exp)

print(f"AVM: {avm:.3f}")
print(f"MAVM: D_PLUS={mavm.d_plus:.3f}, D_MINUS={mavm.d_minus:.3f}")
print(f"KS: {ks:.3f}")
print(f"CVM: {cvm:.3f}")

# %%
# 3. Compare paired observations
# ------------------------------
#
# Deterministic errors require equal-length samples with matching observations
# at each index. The AVM and RMSE both recover the known two-unit shift here.
errors = vl.calc_deterministic_metrics_1d(sim, exp)
print(f"Mean absolute error: {errors.abs_err:.3f}")
print(f"Relative error: {errors.rel_err:.3f}")
print(f"RMSE: {errors.rmse:.3f}")
print(f"NMSE: {errors.nmse:.3f}")
