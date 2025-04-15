# Developer Guide for `pyvale`

## Coding Languages
All user interfaces in `pyvale` should be written in Python to allow ease of use for the general engineering and scientific community. Code where performance is required should be written in a compiled language. A list of preferred coding languages for `pyvale` is given below:
- Python
- Cython
- C/C++
- Zig

The following can be used for linking compiled code to a python interface:
- Cython
- Pybind

GPU programming must be vendor agnostic. The following can be used for GPU programming:
- [HIP](https://github.com/ROCm/hip)

##  Python coding guide
We have non-software engineers and scientists working on the project so these guidelines are directed at a non-specialist:

- Work in your own 'feature' branch, merge into 'dev' - don't push to main (it should be protected and yell at you)!
- Follow the PEP8 style guide: https://peps.python.org/pep-0008/
- Use descriptive variable names, no single letter variables (double letters for iterators in numpy style are ok) single letter variables for indexs / iterators are ok.
- Use major function first variable names: e.g. `FieldScalar`, `FieldVector` and `FieldTensor` instead of `ScalarField`, `VectorField` and `TensorField`.
- Type hint everything: e.g. `def add_ints(a: int, b: int) -> int:`. This makes your code easier to understand and you have the possibility of compiling things if you need.
- `pylint` is a slow linter but will help you if you have type hinted everything.
- Use guard clauses (if statements) with returns at the top of functions to reduce the number of nested if/else structures.
- Default mutable data types to `None`.
- Use `pathlib` and the `Path` class to manage all file io in preference to manual string handling or the `os` module.
- `numpy` and `scipy` are your friend - avoid for/while loops. Push everything you can down into C. Unless you are writing Cython then loops are ok.
- Minimise dependencies as much as possible.
- Avoid decorators unless absolutely necessary (`@dataclass`,  `@abstractmethod` and `@staticmethod` are examples that are ok)
- Don't use `@property` to hide complicated variable initialisation behind the `.` notation.
- No inheritance unless it is an interface (python ABC or protocol) - use composition / dependency injection. See this [video](https://www.youtube.com/watch?v=hxGOiiR9ZKg&t=3s) and thie [video](https://www.youtube.com/watch?v=J1f5b4vcxCQ&t=2s).
- Only use one layer of abstraction - don't inherit from multiple interfaces and don't use mix-ins.
- For interfaces (abstract base classes) prefix the name of the class with a capital `I` e.g. `ISensor`
- For enumerations prefix the name with a capital `E` so `EGeneratorType`.
- Only use abstraction/interfaces when if/else or switch has at least 3 layers and/or becomes annoying
- Use a mixture of plain functions and classes with methods where and when they make sense.
- Imports requiring many `.`'s are annoying. Use the `__init__().py` to help the user with imports and create useful namespaces using classes with just `@staticmethod`.
- Setup good defaults for variables where possible so that the user can get started with minimal input.
- Prefer dataclasses (`@dataclass`) to dictionaries as they tell the user what parameters are needed and can have sensible defaults.
- When using dataclasses `def __post_init__():` is useful for setting defaults for mutable data types.
- Use classes with `__slots__` as it is more memory efficient, faster and stops member variables being added dynamically. For dataclasses use: `@dataclass(slots=True)`.
- Write docstrings when the code is ready for sharing and use autodocstring to help. For `pyvale` we will use `numpy` style docstrings.
- Write some good quickstart examples so people can easily use your code.
- Use code reviews to help each other and be nice / constructive as we are not all software engineers!



