# Agent testing

Run from the repository root using the project Python environment:

```bash
python scripts/test_agent.py tests/sensorsim
python scripts/test_agent.py tests/render -k conformance
python scripts/test_agent.py tests/render/test_imagetools.py::test_image_crop_analytic
```

With uv, prefix commands with `uv run --no-sync --`. Use the smallest relevant
test selection. With no arguments, the runner discovers all tests under `tests/`.
Normal pytest arguments, markers, fixtures, and discovery rules apply; adding a
suite does not require changing the runner.

Successful runs print one summary line. Failures print test IDs, short tracebacks,
and captured output within a default budget of 200 lines and 24,000 characters.
The runner preserves pytest's exit code, including collection errors and no tests
collected. Warnings are retained in the log; their terminal summary is suppressed.

On failure, inspect the printed `Full log` or structured `Reports` path only as
needed. Artifacts are retained in a unique temporary directory. Native output and
truncated diagnostics remain in the full log. Crashes and usage errors fall back
to a bounded log tail. Tests are never rerun automatically.

Increase the budget only when required:

```bash
python scripts/test_agent.py --agent-max-lines 400 --agent-max-chars 48000 tests/render
```

Report the selected tests, summary, and relevant failures. Avoid dumping complete
logs into the conversation. Normal pytest and CI reporting are unchanged.
