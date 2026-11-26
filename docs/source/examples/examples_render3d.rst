.. _examples_render3d:

Render 3D
=========

Render synthetic images for digital image correlation and camera calibration.
The Riley examples cover synthetic imaging for uncertainty quantification.
The Blender examples cover scene construction, deformation and stereo cameras.
For texture-coordinate mapping, see :doc:`examples_renderuvs`.

Riley
-----

.. toctree::
   :maxdepth: 1

   render3d/ex1a_riley_quickstart
   render3d/ex1b_riley_procedural_quickstart
   render3d/ex1c_riley_shaderpipe_quickstart
   render3d/ex1d_riley_dic_from_exodus
   render3d/ex1e_riley_stereocal
   render3d/ex1f_riley_diciruq
   render3d/ex1g_riley_cameramodels

Blender: scenes and deformation
-------------------------------

Build a scene and render deformation for 2D DIC.

.. toctree::
   :maxdepth: 1

   render3d/ex2a_blender_scene
   render3d/ex2b_blender_deformation

Blender: stereo and calibration
-------------------------------

Build stereo scenes, deform them and render calibration targets.

.. toctree::
   :maxdepth: 1

   render3d/ex2c_blender_stereo
   render3d/ex2d_blender_calibration
   render3d/ex2e_blender_stereo_deformation
   render3d/ex2f_blender_calibration_target

.. toctree::
   :hidden:

   examples_renderuvs
