.. _dev_guide_python:

Python Coding Guide
===================================

General Guide
-----------------------------------
We have non-software engineers and scientists working on the project so these guidelines are based on non-specialist python knowledge:

- Work in your own 'feature' branch, merge into 'dev' - don't push to main (it should be protected and yell at you)! We use the 'dev' branch as our main staging branch to ensure we are all in sync.
- Follow the `PEP8 style guide <https://peps.python.org/pep-0008/>`_.
- Use descriptive variable names, no single letter variables (double letters for iterators in numpy style are ok) single letter variables for indices / iterators are ok.
- Use major function first variable names: e.g. ``FieldScalar``, ``FieldVector`` and ``FieldTensor`` instead of ``ScalarField``, ``VectorField`` and ``TensorField``.
- Type hint everything: e.g. ``def add_ints(a: int, b: int) -> int:``. This makes your code easier to understand and you have the possibility of compiling things if you need.
- ``pylint`` is a slow linter but will help you if you have type hinted everything. ``Ruff`` is another good option, it is faster but doesn't pick up type hints as well.
- Use guard clauses (if statements) with returns at the top of functions to reduce the number of nested if/else structures.
- Default mutable data types (lists, dicts, objects) to ``None`` and then set them with an if statement guard clause
- Use ``pathlib`` and the ``Path`` class to manage all file io in preference to manual string handling.
- ``numpy`` and ``scipy`` are your friend - avoid for/while loops. Push everything you can down into C. Unless you are writing Cython then loops are great!
- Minimise dependencies as much as possible.
- Avoid decorators unless absolutely necessary (``@dataclass``,  ``@abstractmethod`` and ``@staticmethod`` are examples that are ok)
- Don't use ``@property`` to hide complicated variable initialisation behind the ``.`` notation - in fact just avoid ``@property`` altogether and just use a ``@dataclass`` for data only classes.
- No inheritance unless it is an interface (python abstract base class ``ABC``) - use composition / dependency injection. See this `video on the flaws of inheritance <https://www.youtube.com/watch?v=hxGOiiR9ZKg&t=3s>`_ and this `video on dependency injection <https://www.youtube.com/watch?v=J1f5b4vcxCQ&t=2s>`_.
- Only use one layer of abstraction - don't inherit from multiple interfaces and don't use mix-ins.
- For interfaces (abstract base classes) prefix the name of the class with a capital ``I`` e.g. ``ISensor``
- For enumerations prefix the name with a capital ``E`` so ``EGeneratorType``.
- Only use abstraction/interfaces when if/else or switch has at least 3 layers and/or becomes annoying.
- Use a mixture of plain functions and classes with methods where and when they make sense.
- Imports requiring many ``.``'s are annoying and the user finds the layers hard to remember. Bring everything to the top level so it can be accessed with ``pyvale.``
- Setup good defaults for variables where possible so that the user can get started with minimal input.
- Prefer dataclasses (``@dataclass``) to dictionaries as they tell the user what parameters are needed and can have sensible defaults.
- When using dataclasses ``def __post_init__():`` is useful for setting defaults for mutable data types.
- Use classes with ``__slots__ = ("var1","var2",)`` as it is more memory efficient, faster and stops member variables being added dynamically. For dataclasses use: ``@dataclass(slots=True)``.
- Use code reviews to help each other and be nice / constructive as we are not all software engineers!

Documentation & Examples
-----------------------------------
Each new feature for ``pyvale`` requires documentation and user examples before being merged into the main package. For docstrings and documentation:

- Every function should have accurate type hints
- Use ``numpy`` style docstrings, there are plugins that can automate some of this if you have used type hints correctly
- For ``numpy`` arrays make sure to include the meaning of each axis of the array and the expected shape of the array in the docstring. For example:

Examples should be placed in the 'src/pyvale/examples/modulename' (see `here <https://github.com/Computer-Aided-Validation-Laboratory/pyvale/tree/main/src/pyvale/examples>`_) directory where 'modulename' is the name of the module you have developed.


Testing
-----------------------------------
We use ``pytest`` as our main testing platform. Tests should be pragmatic and cover the following where applicable:

- Specific algorithms (e.g. tensor rotations) and logic
- Regression tests
- Integration tests
- End-to-end tests

Tests do not need to:

- Have 100% code coverage
- Test initialisation

If you find a bug or are fixing a bug please add a test for that bug as part of the fix.



