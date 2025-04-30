Build while in the correct pyvale virtual environment
```
python setup.py build_ext --inplace
```

<!-- And if debugging:
```
CXXFLAGS="-g -O0" python setup.py build_ext --inplace
``` -->


<!-- Install eigen library to `src/rtcpp/rt/Eigen`. Within this folder should be the header files. -->