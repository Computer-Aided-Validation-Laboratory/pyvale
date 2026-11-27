# Agent Instructions — PyVale

PyVale is a Python-first scientific engineering library for sensor simulation, DIC, rendering, experimental design, and validation/UQ.

## 1. Development approach

- Follow the user's current task and implementation plan. Do not expand scope without a concrete reason.
- Favour simple, explicit implementations. Avoid speculative abstractions, excessive indirection, unnecessary helpers and backwards-compatibility code.
- PyVale is under active development. Breaking changes are acceptable; prefer clean, consistent APIs over compatibility.
- Expect iterative development and extensive refactoring. Do not preserve existing designs merely because they are implemented.
- **Do not update documentation, examples or unrelated docstrings unless explicitly requested.** Documentation consolidation is a separate development phase.

## 2. Code quality

- Follow the relevant [CAVL style guides](https://github.com/Computer-Aided-Validation-Laboratory/styleguides).
- Prefer user-specified reference modules and examples as the source of existing architectural and coding conventions.
- Keep user-facing APIs clear, typed and Python-first. Use native code for justified performance-critical operations.
- Preserve scientific meaning: units, coordinate conventions, array shapes, numerical assumptions and uncertainty definitions.
- Design numerical code with attention to complexity, memory layout, allocations and data movement.
- Validate inputs at appropriate boundaries; avoid redundant checks in performance-critical kernels.
- Keep Python/native boundaries explicit, including types, ownership, lifetime and memory layout.

## 3. Context and repository exploration

- Minimise unnecessary file reads, tool calls and output.
- **Inspect user-specified reference modules or examples first**, limiting inspection to relevant files and patterns.
- If no references are specified, inspect a narrowly scoped related module or example.
- Begin with files identified by the task. Expand exploration only when a concrete dependency or problem requires it.
- Use targeted searches and read relevant sections rather than entire files or directories.
- Do not repeatedly read unchanged files or documentation already available in context.
- Consult `dev/README.md` and relevant style guides only when needed; do not automatically load them.

## 4. Implementation and refactoring

- Establish the implementation approach from the task specification and relevant reference code before making substantial changes.
- Follow agreed architectural decisions unless a concrete problem justifies revisiting them.
- Prefer coherent batches of changes over repeated small edits and checks.
- Use deterministic transformations for large, mechanical refactors where practical.
- Avoid unrelated clean-up, speculative redesign and unnecessary compatibility shims.
- Do not perform unsolicited polishing or consolidation.

## 5. Testing and verification

- Add or update focused tests when implementing or changing numerical behaviour. Prefer analytic/reference tests, explicit tolerances and representative regression cases.
- **Always use `scripts/test_agent.py` for agent-driven pytest runs.** Do not invoke pytest directly unless the harness is unsuitable for a specific debugging requirement.
- Run the smallest relevant test selection during development. Run broader suites only when appropriate or explicitly requested.
- Run from the repository root using the project Python environment:

```bash
python scripts/test_agent.py tests/sensorsim
python scripts/test_agent.py tests/render -k conformance
python scripts/test_agent.py tests/render/test_imagetools.py::test_image_crop_analytic
```

- With uv, prefix commands with `uv run --no-sync --`.
- The harness accepts normal pytest arguments. Running without test arguments discovers all tests under `tests/`.
- Successful runs produce one summary line. Failures provide bounded diagnostics (default 200 lines / 24,000 characters), with full logs and structured reports retained in a unique temporary directory.
- **Inspect full logs only when the displayed failure details are insufficient.** Never dump complete logs into the conversation.
- Increase diagnostic limits only when necessary, using `--agent-max-lines` and `--agent-max-chars`.
- The harness preserves pytest exit codes and never automatically reruns tests. Do not repeatedly rerun successful checks without a concrete reason.
- Report the selected tests, summary and relevant failures. Never claim tests or benchmarks were run unless they actually were.

## 6. Communication

- Keep responses concise. Avoid narrating routine operations or repeating the implementation plan.
- Raise genuine architectural ambiguities before substantial implementation.
- Summarise changed files, verification results and outstanding issues on completion.
- Stop when the requested work is complete.

## 7. Python Style Guide Summary

These rules summarise the [CAVL Python style guide](https://github.com/Computer-Aided-Validation-Laboratory/styleguides/blob/main/guides/python_style_guide.md). They apply to all Python code. Consult the full guide for details when needed.

### Naming and typing

1. **Functions are verbs:** use descriptive `snake_case` names beginning with an action, e.g. `calculate_error()`, `extract_surface()`. Avoid generic names like `process()`, `handle()` or `run()` without clear context.
2. **Variables are descriptive:** prefer longer, meaningful names over ambiguous ones such as `data`, `values`, `result`, `temp` or single letters. Names should communicate what the variable represents.
3. **Classes are nouns:** use `PascalCase`, with the major concept first for related type families, e.g. `FieldScalar`, `FieldVector`.
4. **Naming conventions:** prefix abstract interfaces with `I` (`ISensor`), enums with `E` (`EInterpolationType`), and use `UPPER_SNAKE_CASE` for constants.
5. **Type hints are mandatory:** annotate all function parameters, return values and public attributes. Avoid `Any` and use modern union syntax (`Thing | None`).
6. **Use enums for finite sets:** never use arbitrary strings or string literals to represent a fixed collection of meaningful options. Prefer explicit `Enum` types over `Literal`.
7. **Avoid redundant booleans:** do not combine an enable/disable flag with an optional configuration or value. Use `Thing | None` where absence represents disabled behaviour. Use enums for mutually exclusive modes.
8. **Make APIs explicit:** use keyword-only configuration arguments, avoid positional booleans, and expose important inputs directly rather than hiding them in unnecessary configuration layers.

### Data structures and architecture

9. **Inheritance is restricted to pure abstract interfaces (`ABC`) only.** Concrete classes must never inherit implementation from other concrete classes. No implementation inheritance, multiple inheritance, mix-ins or inheritance from multiple interfaces.
10. **Prefer composition:** use plain functions, dataclasses, composition and dependency injection. Introduce abstract interfaces only when they solve a genuine problem; keep abstraction to one layer.
11. **Prefer typed dataclasses:** use `@dataclass(slots=True)` for structured data and configuration instead of dictionaries or large tuples. Keep dataclasses primarily concerned with data.
12. **Avoid unnecessary helpers:** do not extract trivial single-use functions, introduce needless indirection or split readable algorithms across excessive layers.
13. **Use explicit control flow:** prefer guard clauses, meaningful intermediate variables and straightforward loops. Avoid complex nested comprehensions and clever one-line expressions.

### Banned and restricted Python features

14. **Banned:** operator overloading, `@property`, metaclasses, monkey patching, custom attribute access (`__getattr__`, `__getattribute__`, `__setattr__`) and double-underscore name mangling.
15. **Banned:** dynamic attribute creation, wildcard imports, `eval()` and `exec()`. Do not invent custom dunder-based object semantics.
16. **Restrict decorators and lambdas:** avoid unnecessary decorators. Use lambdas only for trivial callbacks or sort keys; otherwise use named functions.
17. **Avoid hidden side effects:** never silently mutate caller-owned arguments. Make mutation explicit through returned objects or `out=` parameters. Avoid mutable global state.

### Numerical code and correctness

18. **Design for performance:** consider array layout, memory access, allocations, copies, vectorisation and computational complexity from the beginning.
19. **Prefer NumPy/SciPy:** use vectorised operations over Python loops across numerical elements. Ordinary loops over high-level objects are fine; compiled numerical code may use explicit loops.
20. **Make array meaning explicit:** preserve and identify array shapes, axes, dtypes, coordinate conventions and units. Avoid unnecessary copies and surprising shared-memory behaviour.
21. **Fail explicitly:** reject invalid inputs with clear exceptions. Do not silently recover, guess physical assumptions or use exceptions for normal control flow.
22. **Keep computations separate:** numerical functions must not unexpectedly perform I/O, plotting, printing or other user interaction.

### Testing and conventions

23. **Test behaviour and numerical correctness:** prefer analytic solutions, invariants and independent reference results. Avoid redundant tests and tests coupled unnecessarily to implementation details.
24. **Follow project conventions:** use Ruff (88-character lines), Pyright, `pathlib.Path`, shallow public imports and conservative dependencies. Prefer clear, predictable APIs over clever Python.
