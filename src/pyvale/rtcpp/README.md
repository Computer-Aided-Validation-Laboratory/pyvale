make sure to build using cmake the python module. It should be placed in this very folder. Then python files will run.

To generate syntax highlighting in python, run `stubgen --inspect -m pyray` with the venv activated. Move the generated .pyi file to the same place as the .so file, which should be this very folder.

<!-- Build while in the correct pyvale virtual environment
```
python setup.py build_ext --inplace
```

Install eigen library to `src/rtcpp/rt/Eigen`. Within this folder should be the header files. -->