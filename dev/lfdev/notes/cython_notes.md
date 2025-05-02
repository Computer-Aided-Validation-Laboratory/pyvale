## Getting cython working


## setup.py
```python
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fibonacci.py"),
)
```

## Build
```shell
python setup.py build_ext --inplace
```

## Checking if a module is compiled
In the module:
```python
import cython
if cython.compiled:
    print("Yep, I'm compiled.")
else:
    print("Just a lowly interpreted script.")
```

In the script running the module look at the __file__ repr:
```python
import MODULE

print(MODULE.__file__)
```
It should be the compiled *.so

```cython
# @cython.ccall
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def mesh_grid_2d_para(x: cython.double[:], y: cython.double[:]
#                  ) -> tuple[np.ndarray,np.ndarray]:

#     x_max: cython.size_t = x.shape[0]
#     y_max: cython.size_t = y.shape[0]

#     x_grid = np.empty(shape=(y_max, x_max), dtype=np.float64)
#     y_grid = np.empty(shape=(y_max, x_max), dtype=np.float64)

#     x_grid_view: cython.double[:,:] = x_grid
#     y_grid_view: cython.double[:,:] = y_grid

#     ii: cython.size_t
#     jj: cython.size_t

#     with cython.nogil, parallel():
#         for ii in prange(y_max):
#             for jj in range(x_max):
#                 x_grid_view[ii,jj] = x[jj]
#                 y_grid_view[ii,jj] = y[jj]

#     return (x_grid,y_grid)


# @cython.ccall # python+C or cython.cfunc for C only
# @cython.boundscheck(False) # Turn off array bounds checking
# @cython.wraparound(False)  # Turn off negative indexing
# @cython.cdivision(True)    # Turn off divide by zero check
# def mesh_grid_2d(x: cython.double[:], y: cython.double[:]
#                  ) -> tuple[np.ndarray,np.ndarray]:

#     x_max: cython.size_t = x.shape[0]
#     y_max: cython.size_t = y.shape[0]

#     x_grid = np.empty(shape=(y_max, x_max), dtype=np.float64)
#     y_grid = np.empty(shape=(y_max, x_max), dtype=np.float64)

#     x_grid_view: cython.double[:,:] = x_grid
#     y_grid_view: cython.double[:,:] = y_grid

#     ii: cython.size_t
#     jj: cython.size_t

#     for ii in range(y_max):
#         for jj in range(x_max):
#             x_grid_view[ii,jj] = x[jj]
#             y_grid_view[ii,jj] = y[jj]

#     return (x_grid,y_grid)

```