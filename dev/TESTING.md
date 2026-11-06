# Testing Guide

Run the full test suite from the repository root with:

```bash
python -m pytest
```

If the project environment is managed with uv, use the equivalent command:

```bash
uv run --no-sync -- python -m pytest
```

Run one test module or directory by passing its path. The main test areas are
listed below.

| Area | Command |
| --- | --- |
| Dataset helpers | `python -m pytest tests/data` |
| Data I/O | `python -m pytest tests/dataio` |
| DIC | `python -m pytest tests/dic` |
| Rabbit/MOOSE and Gmsh integration | `python -m pytest tests/mooseherder` |
| Rendering | `python -m pytest tests/render` |
| Sensor simulation | `python -m pytest tests/sensorsim` |
| Strain | `python -m pytest tests/strain` |
| Documented examples | `python -m pytest tests/examples` |

Use ordinary pytest selectors for a smaller target, for example
`python -m pytest tests/render/test_pxint2d.py -k newton`.

## Documented examples

Example smoke tests execute examples in a temporary working directory. Their
normal `pyvale-output` files are therefore checked without modifying the
source tree. Run every supported example with:

```bash
python -m pytest tests/examples
```

Use `--example-module` to run examples in one example directory. The value is
the directory below `src/pyvale/examples`, for example:

```bash
python -m pytest tests/examples --example-module=render3d
python -m pytest tests/examples --example-module=render2d
python -m pytest tests/examples --example-module=dic
python -m pytest tests/examples --example-module=basicsensorsim
python -m pytest tests/examples --example-module=extsensorsim
python -m pytest tests/mooseherder tests/examples --example-module=mooseherder
```

Run render implementation tests and its examples together with:

```bash
python -m pytest tests/render tests/examples --example-module=render3d
```

The `example_slow` marker labels longer tests; select it explicitly with
`python -m pytest tests/examples -m example_slow`. It is not excluded from a
normal example run.

To add an example test, place the runnable example below
`src/pyvale/examples/<module>/`, then add it to the appropriate parameter list
in `tests/examples/test_<module>_examples.py`. Use the `run_example` fixture
from `tests/examples/conftest.py`, declare expected output paths, and provide
any input files through `support_files`. For a non-parameterized example test,
add `@pytest.mark.example_module("<module>")` so the module filter includes it.

## Gold regression data

Generate gold data only after deliberately reviewing a known-good result. Gold
is a committed regression baseline, not test output to refresh after a failure.
Run generators from the repository root and inspect the resulting diff before
committing it.

Generators belong in `scripts/`, even when they support a single test module.
Generated files belong beside the tests that consume them:

| Area | Generator | Gold location |
| --- | --- | --- |
| Render conformance | `python scripts/gengold_render_conformance.py --write` | `tests/render/gold_<backend>/` |
| PixInt2D render | `python scripts/gengold_pxint2d.py --write` | `tests/render/gold_pxint2d/` |
| Riley render | `python scripts/gengold_riley_rabbits.py --write` | `tests/render/gold_riley/` |
| Blender triangle | `python scripts/gengold_blender_triangle.py` | `tests/render/gold_blender/` |
| Legacy Blender | `python scripts/gengold_blender.py` | `tests/render/gold_blender/2D_gold/`, `tests/render/gold_blender/3D_gold/` |
| Simulation text I/O | `python scripts/gengold_sim_txt.py` | `tests/dataio/txt_gold/` |
| Sensor simulation | `python scripts/gengold_sensorsim_scalar.py` | `tests/sensorsim/gold/` |

The vector, tensor, and combined sensor-simulation generators are
`gengold_sensorsim_vector.py`, `gengold_sensorsim_tensor.py`, and
`gengold_sensorsim_all.py`. They follow the same convention. MOOSE herd output
is committed under `tests/mooseherder/output_gold/`; no dedicated generator is
currently provided, so update it only from an independently reviewed run.

## Module-specific conditions and skips

### Blender

Blender rendering integration tests and legacy Blender regression tests skip
unless Blender is available. Availability requires Python 3.13 and the optional
`pyvale[blender]` dependency. Blender rendering examples use the same
condition. The availability boundary tests still run without Blender and
verify the diagnostic behaviour.

The Blender adapter is verified for Tri3 input. It warns when given another
render surface topology, then tessellates that surface to Tri3 for the legacy
Blender scene path.

### MOOSE and Gmsh

Rabbit FEM and Python Gmsh are core dependencies. The mooseherder suite runs real
SMP solves, both mesh input methods, nested/ordered decks, analytic mechanics,
parallel chains and every tutorial in CI; no local MOOSE/Proteus checkout is
needed. Install the system GLU library on Linux (`libglu1-mesa` on Ubuntu).

MPI command assembly and rejection of a serial build are tested without MPI.
Actual multi-rank runs need an MPI-capable Rabbit/custom app and compatible
launcher; examples default to SMP even in the MPI tutorial. Select MPI inputs
explicitly. Custom executable behaviour is tested independently of user installs.

Run and build only the mooseherder gallery with:

```bash
python -m pytest tests/mooseherder tests/examples --example-module=mooseherder
PYVALE_DOCS_RUN_EXAMPLES=1 PYVALE_DOCS_EXAMPLE_MODULE=mooseherder python -m sphinx -b html docs/source docs/build/html
```

All 36 historical simulation cases have matching `_mpi.i` entries. Tests mesh
all historical geometries and parse both entry variants. Representative SMP
cases are solved in full. Existing Exodus baselines remain unchanged; regenerate
only deliberately reviewed results with `scripts/run_example_simcases.py`.

### Examples

The example suite skips Blender-backed examples when Blender is unavailable.
It intentionally does not include DIC examples requiring interaction or
unbundled local files: `ex01_region_of_interest.py`, `ex06_hrdic.py`,
`ex08_calibration.py`, `ex09_stereo.py`, `ex10_stereo_platehole.py`, and
`ex11_dic_chal.py`. The `valid` example module is also excluded while it is
under active development.
