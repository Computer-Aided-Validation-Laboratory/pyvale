# Documented example tests

This directory smoke-tests the stable Sensorsim, render, and DIC galleries in
isolated subprocesses. Each test uses a temporary working directory, so the
examples write their normal `pyvale-output` directory without modifying the
repository.

Run all documented example tests with:

```bash
python -m pytest tests/examples
```

Run the examples belonging to one sub-module, such as render, with:

```bash
python -m pytest tests/examples --example-module=render
```

Run the render unit tests and only render examples together with:

```bash
python -m pytest tests/render tests/examples --example-module=render
```

The Blender examples are included but skip automatically when its optional
backend is unavailable. Tests marked `example_slow` remain part of the normal
test set and can be selected with `-m example_slow` when needed.

The suite intentionally excludes DIC examples that need user interaction or
unbundled local input: `ex01_region_of_interest.py`, `ex06_hrdic.py`,
`ex08_calibration.py`, `ex09_stereo.py`, `ex10_stereo_platehole.py`, and
`ex11_dic_chal.py`. Modules still under active development, including `valid`,
are not included.
