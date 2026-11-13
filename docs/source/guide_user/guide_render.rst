.. _guide_render:

Render for DIC uncertainty studies
======================================

Create images with known motion, analyse them with DIC and investigate how
camera and analysis choices affect the recovered measurements.

Before you start
--------------------

Use Python 3.11+ for Riley. Blender is implemented and optional: use Python
3.13 with ``pip install "pyvale[blender]"``. Start with a reference/deformed
image pair before attempting a campaign. Keep an independent record of the
imposed motion so you can assess the recovered displacement.

1. Choose the image formation model
---------------------------------------

Rasterisation projects a surface onto the camera and fills the covered pixels
from the surface pattern. Riley is PyVale's default renderer for DIC studies:
it lets you explore texture scale, camera geometry, distortion and blur without
tracing many paths through a lighting scene.

Path tracing samples possible light paths between the camera, surfaces and
lights. Blender's Cycles engine supports studies where illumination, reflections
or transmission matter. More light samples reduce rendering noise but cost
computation. Do not mistake rendering noise for the camera noise you intend to
simulate.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Approach
     - Useful for DIC UQ
     - Main tradeoff
   * - Rasterisation (Riley)
     - Large image sets with controlled geometry, speckles and camera effects.
     - Fast projection and sampling; it does not model complete light transport.
   * - Path tracing (Blender Cycles)
     - Studies where reflections, shadows or transmission affect the images.
     - More optical detail, but more computation and sampling noise to control.

Feebee is PyVale's own path tracer under development, intended to support
higher-order elements and nested dielectrics for complex optical scenarios.
Its PyVale integration is not yet a runnable alternative to Riley or Blender.

2. Describe a surface and its pattern
-----------------------------------------

A **surface mesh** describes the specimen geometry and its motion. A
**camera** defines the projection onto pixels. A **scene** groups the surfaces,
cameras and any lights required by the backend; the **renderer** turns that
scene into images.

A **texture** is an image, such as a speckle pattern, placed on the surface.
A **shader** is the rule that determines appearance at a surface point: it may
read a texture, calculate a procedural pattern or evaluate a lighting/material
model. Riley and Blender share scene concepts but use their own shader and
material definitions; a scene is not automatically interchangeable between
backends.

**UV coordinates** are two coordinates attached to the surface that locate
points in a texture image. They connect a physical point on the specimen with
a point in the speckle pattern. Keep that mapping attached to the material as
it moves. UV scale, pattern feature size and camera magnification together
determine the speckle size in image pixels.

3. Render a first image
---------------------------

``Mesh3D.coords`` stores reference positions as ``(nodes, 3)`` and its
connectivity uses one element per row with zero-based node indices. Optional
motion has shape ``(frames, nodes, 3)``. Geometry, physical pixel size and focal
length must use the same length unit.

This small checkerboard scene tests the mesh → camera → renderer path. After
it runs, use the Exodus DIC example below to add the displacement sequence and
speckle texture needed for a deformation study.

.. code-block:: python

   from pathlib import Path

   import numpy as np
   import riley
   from scipy.spatial.transform import Rotation
   from pyvale import render

   # A small calibration scene: world lengths use one consistent unit.
   mesh = render.Mesh3D(
       element_type=render.EElemType.TRI3,
       coords=np.array(((-1., -1., 0.), (1., -1., 0.), (0., 1., 0.))),
       connectivity=np.array(((0, 1, 2),)),
       shader=riley.FunctionShader(
           builtin=riley.FuncShaderBuiltin.checker,
           coord_mode=riley.FuncCoordMode.world_reference,
           params=riley.FuncShaderParams(coord_scale=(4., 4.)),
       ),
   )
   camera = render.Camera(
       pixels_num=np.array((256, 256)),
       pixels_size=np.array((0.02, 0.02)),
       pos_world=np.array((0., 0., 2.)),
       rot_world=Rotation.identity(),
       roi_cent_world=np.zeros(3),
       focal_length=1.,
       subsample=2,
   )
   camera = render.cam_frame_mesh(camera, mesh, fov_scale=1.)
   config = riley.RasterConfig(
       parallel=1,
       save_strategy=riley.SaveStrategy.both,
       image_save_mode=riley.ImageSaveMode.grey,
   )
   renderer = render.Riley(config, Path.cwd() / "pyvale-output/guide_render")
   result = renderer.render(render.Scene3D([mesh], [camera]))
   print(result.images.shape)  # frames, cameras, rows, columns, channels

Analyse the result
------------------

Inspect the framing, visible surface and pattern scale before correlating
images. ``Camera.pixels_num`` uses ``(width, height)``; in-memory image stacks
use ``(frames, cameras, height, width, channels)``. File outputs are listed in
``RenderResult.output_paths``; ``images`` may be ``None`` for file-only output.

Establish a rigid-motion baseline and compare DIC measurements with the known
motion in the appropriate coordinate system. Perspective and visibility affect
the relationship between nodal displacements and image measurements. For stereo
work, keep the camera and world coordinate conventions explicit.

Increase subpixel sampling until its effect on the measurement is acceptably
small. If using path tracing, assess light-sampling noise separately. Then vary
the experimental factor of interest and compare measurement bias and variation,
keeping the known motion and render settings with each result.

Continue with these examples and guides:

:doc:`Riley quickstart </examples/render3d/ex1a_riley_quickstart>` · :doc:`DIC images from Exodus </examples/render3d/ex1e_riley_dic_from_exodus>` · :doc:`UV examples </examples/examples_renderuvs>` · :doc:`Blender examples </examples/examples_render3d>`

:doc:`Load simulation data <guide_simdata>` · :doc:`Build meshes <guide_meshing>`
