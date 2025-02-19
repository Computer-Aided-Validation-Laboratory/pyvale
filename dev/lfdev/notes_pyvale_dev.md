
# Notes: `pyvale` developement

## Ideas for papers:
- FUSION SIM: "A probabilistic analysis of residual stress in divertor monoblocks" - showcase mooseherder
- EXP SIM: "Are point wise validation metrics suitable for image-based data?"
- EXP SIM: "A comparison of image-based and point-wise validation metrics for DIC"
- EXP SIM: "A comparison of image rendering techniques for DIC UQ"
- EXP SIM: "Camera placement optimisation for 2D DIC FEMU"
- EXP SIM: "A rendering engine for UQ of IRT sensors"

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

## TODO: `CameraRaster`
- Speed up edge function calculation using stepwise optimisation on SAP
- Try to setup tiling optimisation
- Deal with quads: edge function and interpolation

`CameraRay`
- Build a ray casting version. Only need primary rays.
- Interpolation can be done in world coords using primary ray intersection.
- Still have the problem of dealing with quads


## TODO: `pyvale`
- TODO PRIORITY:
    - Docstrings
    - Tests
    - Field errors assume all sensors sample at the same time but it should be possible to have all sensors sampling at different times.
    - Support for surface mesh extraction to simplify projections

- BUGS!
    - Spatial averaging with rectangle or quadrature makes assumptions about sensor orientation - looks like it assumes XY orientations only. Check this.
    - Should be able to fix this with a good old 4x4 sensor_to_world matrix.

- TODO GENERAL:
    - Build Rory's simple DIC strain filter on top of the basic camera
    - Visualisation tools for perturbed field errors:
        - Angle
    - Visualisation tools for animating sensor traces
    - Visualisation tools for subplots of multiple sensors?
    - Finish basic camera

- TESTING/FEATURE EXAMPLES:
- TODO: ErrorIntegrator
    - Simplify the memory efficient and non-memory efficient options

- TODO: visualisation tools for:
    - TODO: remove methods from sensor descriptor data class
    - TODO: presentation animations - animate traces
    - TODO: experiment - allow extraction of different conditions for comparison plots
    - TODO: visualise all components of vector / tensor field
        - See this example for subplots: https://docs.pyvista.org/examples/02-plot/cmap#sphx-glr-examples-02-plot-cmap-py

- TODO: Field based errors:
    - TODO: Temporal averaging error
        - Set an integration time
        - Set a weighting function

- IMAGE DEF: allow upsampled image to be generated once and retained.

- CAMERAS:
    - Need CameraData class
    - Create 'CameraBasic' class or Simple?
    - Create 'CameraProjection' class
    - Create 'CameraRayTracing' class
    - Create 'CameraIRThermo'
    - Create 'CameraDIC2D'
    - Create 'CameraDICStereo'
    - Allow field class to do a single field rotation into camera coords on creation and store this
    - Allow all sensors to have

- TESTING:
    - Need to check rotations are consistent

## Sensors
A sensor should have:
- A spatial measurement geometry: point, line, area, volume
- A measurement position, centroid of the measurement geometry
- A measurement point/area/line/volume
- A measurement frequency in time
- A calibration curve
- A digitisation

### Thermocouples
https://www.mstarlabs.com/sensors/thermocouple-calibration.html

T  =  -0.01897 + 25.41881 V - 0.42456 V^2 + 0.04365 V^3
where V is voltage in units of millivolts and T is temperature in degrees C.

- Thermocouple amplifier card:
https://www.ni.com/docs/en-US/bundle/ni-9213-specs/page/specs.html

- App note on thermocouple amplifier chip:
https://www.analog.com/en/resources/app-notes/an-1087.html

## Systematic errors
Comes from many sources:
- Positioning error
- Averaging over time (integration time)
- Averaging over space (integration space)
- Digistisation error
- Calibration error
- Temporal lag of the signal
- Drift


## Random errors
Comes from sensors noise, need a way to specify a probability distribution to sample from.

## Numba
https://www.youtube.com/watch?v=6oXedk2tGfk

## Gauss Quadrature

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
