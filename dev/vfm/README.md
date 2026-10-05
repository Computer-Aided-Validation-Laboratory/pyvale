# VFM development utility

`prepare_assembled_input.py` is the retained external-data preparation
example. It converts a solver-independent directory of assembled NumPy arrays
into PyVale's persisted `ExperimentData` format.

```bash
python dev/vfm/prepare_assembled_input.py RAW_DATA PREPARED_DATA \
    --thickness-mm 0.8 --loading tensile-x
```

The raw directory requires `x.npy`, `y.npy`, `strain.npy`, `force.npy`, and
`time.npy`. It may also contain `specimen_mask.npy`,
`known_parameter_maps.npz`, and `metadata.yaml`.
