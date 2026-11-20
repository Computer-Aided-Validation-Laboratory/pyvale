.. _examples_render3d:

Render 3D
=========

Render synthetic images for digital image correlation and camera calibration.
The Riley examples cover PyVale's renderer, imaging effects and feature options;
the Blender examples cover scene construction, deformation and stereo cameras.
For texture-coordinate mapping, see :doc:`examples_renderuvs`.

Riley: getting started
----------------------

Create scenes and render speckled surfaces with different materials.

.. toctree::
   :maxdepth: 1

   render3d/ex1a_riley_quickstart
   render3d/ex1b_riley_sphere200
   render3d/ex1c_riley_rabbits

Riley: DIC and calibration
--------------------------

Render DIC sequences from simulations and construct stereo calibration targets.

.. toctree::
   :maxdepth: 1

   render3d/ex1d_riley_dicuq
   render3d/ex1e_riley_dic_from_exodus
   render3d/ex1f_riley_stereocal

Riley: imaging effects and features
-----------------------------------

Explore point spread functions and the renderer feature collection.

.. toctree::
   :maxdepth: 1

   render3d/ex1g_riley_psf
   render3d/ex1h_riley_feature_zoo

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
