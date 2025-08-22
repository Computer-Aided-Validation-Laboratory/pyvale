# Notes: `pyvale` developement




## TODO: `pyvale` point sensors
--------------------------------------------------------------------------------
- **TESTS**
    - Tests for vector and tensor rotations
    - Tests for analytic fields and field integration
    - End-end tests based on examples


- EXPERIMENT ERROR ANALYSIS MODULE / UQ MODULE
    - Extracts which errors in the chain are contributing the most for each sensor
    - Show differences between simulations for experiment traces

- BUGS!
    - Node vs Elem vars in the SimData object - does pyvale work with elem vars???
    - Spatial averaging with rectangle or quadrature makes assumptions about sensor orientation - looks like it assumes XY orientations only. Check this.
    - Should be able to fix this with a good old 4x4 sensor_to_world matrix.

- TODO PRIORITY:
    - Tests
    - Field errors assume all sensors sample at the same time but it should be possible to have all sensors sampling at different times.
- Area/volume/line averaging:
    - Fix orientation of spatial averaging points with sensor_to_world matrix
    - Support different types of spatial averaging and different shapes
- Field based errors:
    - Temporal averaging error:
        - Set an integration time
        - Set a weighting function

- ErrorIntegrator
    - Simplify the memory efficient and non-memory efficient options
- TESTING:
    - Need to check rotations are consistent

- SENSORFACTORY
    - Add more typical sensors to the library
    - Add extensometers
    - Add strain gauges with arbitrary rosette dims


## TODO: `Raster`
--------------------------------------------------------------------------------
- Speed up edge function calculation using stepwise optimisation on SAP
- Try to setup tiling optimisation
- Deal with quads: edge function and interpolation

**NOTE**: should we just have a single `IRenderer` or `IImager` interface and unify everything including: Blender, rasteriser, ray tracer?


### `Renderer` Architecture
- Meshes without and without deformation in the same scene:
LOOP: over meshes
    - raster_static_mesh() - takes one mesh
        - Input/output: image buffer, depth buffer
    - raster_deform_mesh() - takes one mesh
        - Input/output: image buffer, depth buffer

- What do we do when the number of time steps in the render or displacement fields does not match between meshes?
    - Need to set default texture/field of 0
    - Need to then mask with np.nan?
    - How do we tell what is zero and what is background?
- Deal with rendering at non-simulation time-steps


- Start with an image buffer and depth buffer
    - Render first mesh
    - Pass in next mesh and the previous image buffer and depth buffer
    - Repeat until all meshes are rendered

- **COORDS**: Joined coord table: has the same shape
- **CONNECT**: Separate connectivity tables (support quad/tri) indexing into the coords
    - Will have the same shape for triangles only
    - Will not have the same shape for different element types
- **FIELDS**: How do we deal with render and displacement fields?
    - Will not have the same shape
    - All meshes will need to have the same number of field to render
        - If not create a dummy field of zero
        - How do we know which field is which?


**OPTIONS**
- The user will only add a camera if they want to render it
- Should we parallelise over cameras even for a single frame?


`Renderer`
    - `__init__()` takes what? - an `IRenderer`: `Blender`, `Raster`, `RayTrace` which has a
        - `RenderOpts` unique to the IRenderer
    - Should implement the function that is called for parallelisation
    - Should implement the function that saves the image so it is consistent
    - Should implement the multi-processing parallelisation by frame?
    - `render_one_frame`
    - `render_all_frames(para_by_frame: int | None = None)`
    - Takes a `Scene`
        - `__init__()` - does this need to know what type of renderer we are using
        - `add_obj()`
        - `add_light()`
        - `add_camera()`
        - Has `list[Object]`, `list[Light]`, `list[Camera]`

- Will need wrapper classes around each renderer
- `RasterNP`
- `RasterCY` - python interface on top of cython
    - Need a python wrapper function
    - `rastercyth`

- `Renderer` parallelisation:
    - Should the engine deal with this? This would allow frame by frame or camera by camera parallelisation on an individual engine basis.
    - Python parallel harness would be nice: already needed for raster in numpy/cython
    - Also need to allow parallel in native C/C++ and Zig


### `Raster` Core/Interface
- Manages parallelisation?
- Need to split out parallelisation from numpy version
- Numpy/Cython using `multiprocessing.pool`
- Zig will have it's own parallelisation

### `Raster` Numpy
- Need to be able to actually save images as *.tiff/*.bmp greyscale/colour or other format
- Fix the cython vs no cython image averaging looking at *.so

### `Raster` Cython
- Saving and parallelisation will be managed by `RasterCore`
- Render all single frame / render all frames
- Implement static/deformed meshes

### `Raster` Zig
- Continue building it to match the Cython version


## Cython Packaging:
https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#basic-setup-py

[tool.setuptools]
ext-modules = [
  {name = "example", sources = ["example.pyx"]}
]

[tool.setuptools]
ext-modules = [
    {name = "pyvale.cython.rastercyth", sources = ["src/pyvale/core/cython/rastercyth.py"]}
]

## Validation Metrics:
- Multi-physics
- Multi-fidelity sensing
- Hidden vs non hidden key point: max temp vs imaging a hole with DIC
    - Inference methods / GPs?


## Ideas for papers:
- FUSION SIM: "A probabilistic analysis of residual stress in divertor monoblocks" - showcase mooseherder
- EXP SIM: "Are point wise validation metrics suitable for image-based data?"
- EXP SIM: "A comparison of image-based and point-wise validation metrics for DIC"
- EXP SIM: "A comparison of image rendering techniques for DIC UQ"
- EXP SIM: "Camera placement optimisation for 2D DIC FEMU"
- EXP SIM: "A rendering engine for UQ of IRT sensors"
#

## Thermocouples
https://www.mstarlabs.com/sensors/thermocouple-calibration.html

T  =  -0.01897 + 25.41881 V - 0.42456 V^2 + 0.04365 V^3
where V is voltage in units of millivolts and T is temperature in degrees C.

- Thermocouple amplifier card:
https://www.ni.com/docs/en-US/bundle/ni-9213-specs/page/specs.html

- App note on thermocouple amplifier chip:
https://www.analog.com/en/resources/app-notes/an-1087.html



## Gauss Quadrature: Change of Interval
https://stackoverflow.com/questions/33457880/different-intervals-for-gauss-legendre-quadrature-in-numpy

To change the interval, translate the x values from [-1, 1] to [a, b] using, say,

t = 0.5*(x + 1)*(b - a) + a

and then scale the quadrature formula by (b - a)/2:

gauss = sum(w * f(t)) * 0.5*(b - a)

Gauss Quadrature for the Unit Disc
http://www.holoborodko.com/pavel/numerical-methods/numerical-integration/cubature-formulas-for-the-unit-disk/



## Pyvista Cameras
Tested on monoblock sim:
cpos = xy
[(0.0, 16.0, 90.80825912395183),
    (0.0, 16.0, 5.5),
    (0.0, 1.0, 0.0)]

then, azimuth = 45
[(60.32204851776551, 16.0, 65.8220485177655),
(0.0, 16.0, 5.5),
(0.0, 1.0, 0.0)]

then, zoom = 0.5
[(60.32204851776551, 16.0, 65.8220485177655),
(0.0, 16.0, 5.5),
(0.0, 1.0, 0.0)]

Start with xy then azimuth 90
[(85.30825912395183, 16.0, 5.5000000000000195),
(0.0, 16.0, 5.5),
(0.0, 1.0, 0.0)]



## Memory Profiling with `mprof`
Install into a virtual environment:
`pip install memory-profiler`

Run a script to profile the memory (output is stored in a time stamped dat file in the working directory):
mprof run --python PATH/TO/MAIN.py

Plot the output and save to png:
mprof plot -o memory_profile.png



## ARCHIVE
-------------------------------------------------------------------
# HOW TO: 2D DIC
- Start with a simple pixel wise DIC algorithm to get a starting point
- Start with pure numpy and scipy version then build own interp/opt

- Speckle generator:
    - Allow noise, gaussian blurring, digitisation, format saving
- 2D image deformation:
    - Update existing 2D image deformation in pyvale to use pyvista to do interp
    - Update masking to remove alpha shape - use the edge function
    - Simplify code and make core to pyvale
    - Generate test cases for 640x480 images on rectangular ROIs
- Shape functions:
    - Start with rigid
    - Then add affine
- Correlation criteria:
    - Implement all the different correlation criteria with good pre-calcs
- Optimisers:
    - Start with Nelder-Mead
    - Implement Levenberg-Marquadt
    - Need to return the residual
- Interpolation
    - Use scipy to do spline interp on image
    - Build own spline interp in cython
- Generate a test case run in 2D through DICE

**2D DIC TEST CASE**
- 1020x520 pixels
- 100x50mm plate
- 10px on the border
- Resolution = 100mm/1000px = 0.1 mm/px
- 10px per mm
- 1mm displacement = 10 px
- Need displacement cases at 0.1/10 = 1/10th of a pixel


## EULER ANGLES: Intrinsic vs Extrinsic

**INTRINSIC** = Rotate about localc coords
If the intrinsic Euler angles are α,β,γ representing rotations about the initial X, then the new Y', and then the newest Z'' axes respectively, the final rotation matrix is obtained by multiplying the individual rotation matrices in the order of application. For example, if the sequence is z-y'-x'', the rotation matrix R would be R=z(α)Ry′(β)Rx′′(γ). Here, the primes indicate the axes after the previous rotation. When expressed in terms of the initial fixed frame, this becomes R=Rz(α)Ry(β)Rx(γ). Notice the direct correspondence between the order of application and the matrix multiplication.

**EXTRINSIC** = Rotate about fixed global coords
If the extrinsic Euler angles are α,β,γ representing rotations about the x, y, and z axes respectively, the final rotation matrix is obtained by multiplying the individual rotation matrices in the order they are applied. For example, if the sequence is z-y-x, the rotation matrix R would be R=Rx(γ)Ry(β)Rz(α). Notice the reversed order of application in the matrix multiplication compared to the order of rotations.