# Minimal sensitivity-gated EGI identification

Date: 30 August 2026  
Status: implemented as an optional path; noise calibration remains a scientific gate

## Purpose

Preserve the existing data-driven/SVD toolkit, but test a smaller algorithm that does not create a second inverse problem to configure the identification objective.

## Algorithm

1. Run the existing homogeneous SBVF identification for yield strength and hardening modulus.
2. Evaluate a physical bank of odd EGI windows from three datapoints to half the smaller specimen bounding-box dimension.
3. For each window, calculate the RMS of the strongest configured fraction of the homogeneous-state, normalised EGI observations after dividing by that support's propagated noise.
4. Retain windows passing the signal-to-noise and coverage gates. Select the smallest as fine, largest as broad, and the eligible window nearest their logarithmic midpoint as middle.
5. At the first fixed-BF solve, perturb the global yield-strength map once and the global hardening map once. Convert the two stress-history differences to normalised pointwise magnitudes and combine them with a maximum.
6. Convert that combined space-time activity to a smooth frozen gate. After removing activity below a small relative numerical floor, the default ramp spans the minimum to 90th percentile of positive robustly-scaled activity. This retained over 90% of both parameter activities in the offline audit while placing about 90% of active observations in the smooth transition. The same gate is used for fine, middle and broad EGI observations.
7. Minimise a conventional weighted sum:

   `J = (1 - lambda_F - lambda_B) mean(J_f, J_m, J_b) + lambda_F J_FRE + lambda_B J_broad,all`

   Each term is an RMS of noise-normalised residuals. FRE and the full broad-EGI guard are not sensitivity masked.
8. Keep the gate frozen across BF solves by default. An explicit refresh interval is available for later comparison.

## Deliberate omissions

- no local material-probe bank;
- no native-DOF finite-difference Jacobian;
- no SVD projection;
- no Fisher matrix or redundancy optimisation;
- no sensitivity weighting of FRE;
- no automatic refresh before every BF solve.

## Scientific gate before a workstation run

The current scalar EGI noise value is provisional and failed the earlier interpretation audit. The next experiment must first propagate DIC strain noise through each selected normalised-EGI operator. A workstation identification should only proceed if at least three candidate supports pass the declared SNR and coverage gates, and the three objective/guard contributions are comparable at controlled perturbations.

## Entry point

Use `--simple-data-driven-objective-config dev/vfm/data/wdbn1_simple_sensitivity_gated_objective_v1_20260830.json` with the existing notched-EBW identification caller. The previous `--data-driven-objective-config` path remains unchanged.
