# Notes: numpy

Avoiding loops by applying a function along an array dimension: `apply_along_dimension`

Random number generation in numpy:
```python
import numpy as np

mu, sigma = 0, 0.1
s = np.random.default_rng().normal(mu, sigma, 1000)

s = np.random.default_rng().uniform(-1,0,1000)
```

# Notes: pyvista

## Mesh structure in pyvista

NOTE: pyvista element numbers are 0 indexed so need to -1 from whole element connectivity table to correct it.

The most general mesh structure in `pyvista` is the `pyvista.UnstructuredGrid`. This requires 3 types of information to construct. Here we assume we have a mesh with `N` nodes and `E` elements with `n_per_e` nodes per element which:
 1. A flattened connectivity array (`cells`) of the form `[n_per_e,n_1,n_2,n_3,...]` repeated for each element.
 2. A flattened array that specifies the element types (`cell_types`) as an array of length `E` with the `py.CellType` as the value for each element.
 3. An array specifying the nodal coordinates as an `Nx3` array of nodal coordinates.

Example multi element mesh:
```python
import numpy as np
import pyvista as pv
from pyvista import CellType

pv.set_plot_theme("document")

# Define points for the hexahedron and tetrahedron
points = np.array(
    [
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
        [0.75, 0.5, 1.25],  # 8
    ],
    dtype=float,
)

# Define hexahedral cell connectivity
hex_cell = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7], np.int64)

# Define tetrahedral cell connectivity
tet_cell = np.array([4, 4, 5, 6, 8], np.int64)

# Combine cell arrays
cells = np.hstack((hex_cell, tet_cell))

# Define cell types
cell_types = np.array([CellType.HEXAHEDRON, CellType.TETRA])
print(cell_types)

# Create unstructured grid
grid = pv.UnstructuredGrid(cells, cell_types, points)

# Visualize the unstructured grid
pl = pv.Plotter()
pl.add_mesh(grid, show_edges=True)
pl.show_axes()
pl.show()
```

## Plotting in pyvista

https://docs.pyvista.org/version/stable/examples/02-plot/orbit#sphx-glr-examples-02-plot-orbit-py

https://docs.pyvista.org/version/stable/api/core/camera#cameras-api

Adding multiple meshes to one plot:

```python
import pyvista as pv

pv.set_plot_theme('dark')
pv_plot = pv.Plotter(window_size=[1000, 1000]) # type: ignore
pv_plot.add_mesh(pv_sensdata,
                    label='sensors',
                    color='red',
                    render_points_as_spheres=True,
                    point_size=20
                    )

pv_plot.add_mesh(pv_simdata,
                    scalars=pv_simdata['T'][:,-1],
                    label='sim data',
                    show_edges=True)
pv_plot.camera_position = 'xy'
pv_plot.show(cpos="xy")
```

## Misc pyvista plotting commands

```python
import pyvista as pv

pv.set_plot_theme('dark')

pv_plot = pv.Plotter(window_size=[1280, 800],off_screen=True)
pv_plot.add_axes_at_origin(labels_off=False)
pv_plot.set_scale(xscale = 100, yscale = 100, zscale = 100)
pv_plot.camera_position = 'xy'
pv_plot.camera.zoom(5)
pv_plot.show(cpos="xy")
```

## Labels on pyvista plots

https://tutorial.pyvista.org/tutorial/03_figures/bonus/e_labels.html

```python
poly = pv.PolyData(np.random.rand(10, 3))
poly["My Labels"] = [f"Label {i}" for i in range(poly.n_points)]

plotter = pv.Plotter()
plotter.add_point_labels(poly, "My Labels", point_size=20, font_size=36)
plotter.show()
```

## Scalar bars on plots

https://docs.pyvista.org/version/stable/examples/02-plot/scalar-bars#sphx-glr-examples-02-plot-scalar-bars-py

https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_scalar_bar

```python
pv_plot.add_scalar_bar('Temperature, T [degC]')
```

## Picking from a plot

```python
sphere = pv.Sphere()

pl = pv.Plotter()
pl.add_mesh(sphere, show_edges=True, pickable=True)
pl.enable_element_picking(mode=ElementType.EDGE)

pl.camera_position = [
    (0.7896646029990011, 0.7520805261169909, 0.5148524767495051),
    (-0.014748048334009667, -0.0257133671899262, 0.07194025085895145),
    (-0.26016740957025775, -0.2603941863919363, 0.9297891087180916),
]

pl.show(auto_close=False)
```

```python
mesh = pv.Wavelet()

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, pickable=True)
pl.enable_element_picking(mode=ElementType.FACE)

pl.camera_position = [
    (13.523728057554308, 9.910583926360937, 11.827103195167833),
    (2.229008884793069, -2.782397236304676, 6.84282248642347),
    (-0.17641568583704878, -0.21978122178947299, 0.9594653304520027),
]

pl.show(auto_close=False)

# Programmatically pick a face to make example look nice
try:
    width, height = pl.window_size
    pl.iren._mouse_right_button_press(419, 263)
    pl.iren._mouse_right_button_release()
except AttributeError:
    # ignore this section when manually closing the window
    pass

```

## Interpolating using an FE mesh
https://docs.pyvista.org/version/stable/examples/01-filter/interpolate.html
https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.datasetfilters.sample#pyvista.DataSetFilters.sample

Syntax is:

```python
sampled_values = points_to_sample.sample(fe_field_data_to_sample)
```

## Camera positions for pyvista plots

https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.plotter.camera_position#pyvista.Plotter.camera_position

- Example 1: Plate
```
[(-0.295, 1.235, 3.369),
 (1.0274, 0.314, 0.0211),
 (0.081, 0.969, -0.234)]
```

## Saving images in pyvista

- Option 1: screenshot to png. Need to create a plotter with off_screen=True

```python
import pyvista as pv
pv_plot = pv.Plotter(off_screen=True)
# Add whatever mesh you want
pv_plot.show(screen_shot='save_image.png')
```

- Option 2: export to vector graphics as .svg .eps .ps .pdf .tex

```python
import pyvista as pv
pv_plot = pv.Plotter()
# Add whatever mesh you want
pv_plot.save_graphic('save_image.svg')
```

# Notes: pymoo

## pymoo: GA, Genetic Alogorithm

```python
class GA(GeneticAlgorithm):
    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_and_fitness),
                 crossover=SBX(),
                 mutation=PM(),
                 survival=FitnessSurvival(),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=SingleObjectiveOutput(),
                 **kwargs):

```

## pymoo: PSO, Particle Swarm Optimisation

```python
class PSO(Algorithm):
    def __init__(self,
                 pop_size=25,
                 sampling=LHS(),
                 w=0.9,
                 c1=2.0,
                 c2=2.0,
                 adaptive=True,
                 initial_velocity="random",
                 max_velocity_rate=0.20,
                 pertube_best=True,
                 repair=NoRepair(),
                 output=PSOFuzzyOutput(),
                 **kwargs):
```

##  pymoo: Problem with constraints
```python
class SphereWithConstraint(Problem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=1, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum((x - 0.5) ** 2, axis=1)
        out["G"] = 0.1 - out["F"]
```

## pymoo: Parallelisation:
https://pymoo.org/problems/parallelization.html
Example:
```python
from pymoo.core.problem import Problem

pool = ThreadPool(8)

class MyProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-5, xu=5, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):

        # define the function
        def my_eval(x):
            return (x ** 2).sum()

        # prepare the parameters for the pool
        params = [[X[k]] for k in range(len(X))]

        # calculate the function values in a parallelized manner and wait until done
        F = pool.starmap(my_eval, params)

        # store the function values and return them.
        out["F"] = np.array(F)

problem = MyProblem()
res = minimize(problem, GA(), termination=("n_gen", 200), seed=1)
print('Threads:', res.exec_time)
pool.close()
```


# Notes: Scipy

## Scipy, rotation
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html



# Notes: gmsh

OpenCASCADE Tutorials: 16,18,19,20

## Geometry

**New**
Analogously to ‘newp’, the special
variables ‘newc’, ‘newcl, ‘news’, ‘newsl’ and ‘newv’ select new curve, curve loop, surface, surface loop and volume tags.

## Controlling Mesh Options

**Create a quandrangular mesh from a triangular one**
Recombine Surface{1};

**Control min/max mesh size**
Mesh.MeshSizeMin = 0.001;
Mesh.MeshSizeMax = 0.3;

**Control the mesh algorithm**
Mesh.Algorithm = #

**Creating higher order meshes**
Mesh.ElementOrder = 2;
Mesh.HighOrderOptimize = 2;

**Mesh only part of the model**
- Note change the volume number to make visible
Hide {:}
Recursive Show { Volume{129}; }
Mesh.MeshOnlyVisible=1;

## Controlling Mesh Size
Need to use `Transfinite` functions.

**Control number of nodes on a curve/surface**
- Note: includes nodes at the end of the line.
Transfinite Curve{*line numbers*} = *number of nodes*;

Then use:

Transfinite Surface{*surface numbers*} = {*corner points*}

