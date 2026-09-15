# Merge `render-api` into `dev`

## Scope

Merge `render-api` into `dev` while prioritising the render module and its new
test suites, and retaining the newer stereo-DIC changes from `dev`.

## Conflicts

The merge has 47 conflicted paths:

- 40 binary sensor-simulation gold files in `tests/sensorsim/gold/`
- `.gitignore`, `docs/source/conf.py`, and `pyproject.toml`
- `src/pyvale/examples/dic/ex03_plate_with_hole_strain.py`
- `src/pyvale/examples/dic/ex04_dic_blender.py` (modify/delete)
- `src/pyvale/examples/dic/ex05_dic_challenge.py`
- `tests/strain/test_strain.py`

## Resolution decisions

- Take the `render-api` version of every conflicting `.npy` gold file. These
  are indivisible binary baselines produced for the new test suite.
- Keep the deletion of `ex04_dic_blender.py`. Its render-based replacement is
  `ex04_render_to_dic.py`.
- Combine `.gitignore` rules from both branches, preserving the render output
  and gold-file exclusions as well as DIC and strain temporary-output rules.
- In `docs/source/conf.py`, retain the render galleries and opt-in gallery
  execution configuration, while using the current `dev` release/version.
- In `pyproject.toml`, retain render dependencies and markers (`riley-raster`,
  `numba`, optional Blender extra, and render test markers), use the current
  `dev` version, and do not reintroduce the broad source-distribution exclusion
  of `src/pyvale/data/**`, because render examples need packaged data.
- For DIC examples, retain the `pyvale.data` migration and render output
  conventions, incorporating applicable DIC fixes from `dev`.
- For `tests/strain/test_strain.py`, retain render's `tmp_path` isolation but
  use `print_level`, which is the current DIC/strain argument on `dev`.

## Required post-merge compatibility fixes

The DIC API changes in `dev` auto-merge, but several render-side callers use
superseded argument names:

- In `ex04_render_to_dic.py`, replace both `debug_level=0` calls with
  `print_level=0`.
- In `ex07_incremental.py` and `tests/dic/test_dic.py`, replace
  `incremental=True` with `incremental_update="IMAGE"`, preserving the old
  default incremental behaviour.
- In `tests/strain/test_strain.py`, replace `debug_level=2` with
  `print_level=2` while retaining `tmp_path` output isolation.

## Verification

Run DIC and strain tests first, then the render tests and documented example
tests. Include the stereo-DIC tests/examples: their files auto-merge and must
remain covered even though they are not textual conflicts.
