# Interim EGI resolution workflow

Automatic fine-support selection is not qualified. The supported workflow is:

1. Run `uv run python dev/vfm/egi_resolution_diagnostic.py`.
2. Inspect `EGI_RESOLUTION_DIAGNOSTICS_20260901.pdf` and the complete CSV sweep.
3. Choose the fine EGI window manually in grid points for that dataset.
4. Pass it to identification as `--fine-egi-window N`.
5. Identification derives broad from the configured geometry bank and middle as
   the nearest valid odd support to the logarithmic midpoint.
6. Fine, middle and broad metrics are installed once after Phase 0 and remain
   frozen for the BF trajectory.
7. FRE defaults to a separate Phase-0 noise-propagation sweep. It selects and
   freezes the finest longitudinal slicing that retains at least two rows per
   slice, valid cross-section coverage, correlation P10 >= 0.95 and median
   NRMSE <= 0.25. Use `--force-slices N` only to reproduce a fixed historical
   control; the default is `--force-slices auto`.

Example:

```bash
uv run python dev/vfm/call_notched_ebw_bivariate_identification.py \
  --input /path/to/pyvale-vfm/prepared \
  --simple-data-driven-objective-config \
    dev/vfm/data/wdbn1_simple_sensitivity_gated_objective_v1_20260830.json \
  --fine-egi-window 21
```

The diagnostic noise knee, SNR and map correlations are inspection aids only.
They do not select, warn about, or override the user's fine support.
