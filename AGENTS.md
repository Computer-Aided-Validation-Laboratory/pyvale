# Agent Instructions — PyVale

PyVale is a Python-first scientific engineering library for sensor simulation, digital image correlation, camera rendering, experimental design, and validation/UQ. Keep changes correct, fast, and simple for users.

## Read first

- [Developer guide](dev/README.md), then relevant project documentation under `dev/` and `docs/`.
- [CAVL Python style guide](https://github.com/Computer-Aided-Validation-Laboratory/styleguides/blob/main/guides/python_style_guide.md) — including its Cython section.
- [CAVL Performance Oriented Software guide](https://github.com/Computer-Aided-Validation-Laboratory/styleguides/blob/main/guides/performance_oriented_software_guide.md) — especially for numerical and compiled code.
- [CAVL C++ style guide](https://github.com/Computer-Aided-Validation-Laboratory/styleguides/blob/main/guides/cpp_style_guide.md) — for C/C++ code.
- [CAVL Zig style guide](https://github.com/Computer-Aided-Validation-Laboratory/styleguides/blob/main/guides/zig_style_guide.md) — when touching Zig code.

## PyVale-specific rules

1. **Python-first API.** Keep user-facing configuration and workflows in clear, typed Python; use compiled code for measured or clearly identified performance-critical work.
2. **Scientific meaning matters.** Preserve physical units, coordinate conventions, array shapes and axes, and numerical assumptions. Never silently change their interpretation.
3. **Explicit native boundaries.** Keep Python/native interfaces narrow, with explicit ownership, lifetime, layout and dtype requirements. Follow `dev/README.md` on C ABI and build tooling.
4. **Avoid unnecessary data movement.** Design NumPy arrays and native buffers to minimise allocations, conversions, copies and Python calls inside intensive loops.
5. **Performance is designed in.** Consider computational complexity, data locality, vectorisation and parallelism early; benchmark representative workloads before claiming speedups.
6. **Validate at boundaries.** Reject invalid or ambiguous inputs clearly, and avoid redundant checks inside validated numerical kernels.
7. **Prefer straightforward implementations.** Avoid speculative abstractions, inheritance, extra dependencies and tiny single-use helpers that obscure the algorithm.
8. **API evolution**: Pyvale is a young, evolving project. Do not preserve public APIs at the expense of a cleaner, more consistent design. Breaking changes are acceptable and encouraged when justified. Avoid backwards-compatibility shims, deprecated aliases, and legacy code paths. Update all affected call sites, tests, examples, and documentation.
9. **Verify numerics.** Add focused tests for changes, including analytic/reference comparisons, regression cases and appropriate tolerances; run the relevant existing tests.
10. **Keep documentation aligned.** Update applicable `dev/`, user documentation and examples when changing interfaces, conventions or numerical behaviour; report tests and benchmarks actually run.
