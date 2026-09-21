# Experimental VFM profiling

These files preserve the bounded scalar cost-profile investigation without
making it part of PyVale's supported public API.

- `optimiserscalarprofile.py` scans and refines a scalar cost profile and
  classifies identification evidence such as plateaus and active bounds.
- `optimiserslicewiseprofile.py` applies that diagnostic solve independently
  to each active slice.
- `check_optimiserscalarprofile.py` contains the focused regression checks and
  may be run explicitly with pytest.

The slice-wise prototype imports private helpers from PyVale's established
slice-wise least-squares optimiser. It should therefore be reviewed and
promoted deliberately if it later becomes a supported core optimiser.
