# Virtual Fields Method examples

The release examples are self-contained and use a small analytical tensile
experiment generated in memory:

0. `vfm_ex0_input_data.py`: construct and validate input arrays and specimen data.
1. `vfm_ex1_hom.py`: homogeneous SBVF identification.
2. `vfm_ex2_slicewise.py`: parallel slicewise FRE identification and result
   analysis.
3. `vfm_ex3_slicewise_refinement.py`: slicewise identification with adaptive
   merge/split refinement.
4. `vfm_ex4_hom_slicewise.py`: homogeneous SBVF followed by slicewise FRE.

Run an example directly, for example:

```bash
python src/pyvale/examples/vfm/vfm_ex1_hom.py
```

Examples 2 to 4 have optional `save_results_file` and `save_figures` toggles,
both disabled by default. Saved output is written under
`Path.cwd() / "<example_name>_saved_data"`.
