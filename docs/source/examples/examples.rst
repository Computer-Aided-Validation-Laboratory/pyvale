.. _examples_all:

Examples
========

Choose a module to explore its examples. Each collection starts with a
quickstart and progresses to more detailed workflows.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: SensorSim

      Build virtual sensor arrays, model measurement uncertainty and run
      repeated experiments. Start with the basics, then explore custom data
      and more detailed error models.

      +++

      :doc:`Basic examples <examples_basics_sensorsim>` ·
      :doc:`Extended examples <examples_ext_sensorsim>`

   .. grid-item-card:: Digital Image Correlation

      Define regions of interest, analyse 2D and stereo images, and calculate
      displacement and strain.

      +++

      :doc:`DIC examples <examples_dic>`

   .. grid-item-card:: Render

      Generate synthetic camera images with Riley or Blender. Configure
      scenes, cameras and imaging effects, and control texture mapping.

      +++

      :doc:`Rendering examples <examples_render3d>` ·
      :doc:`UV mapping examples <examples_renderuvs>`

   .. grid-item-card:: Validation

      Compare simulation predictions with experimental measurements and
      analyse uncertainty in validation metrics.

      +++

      :doc:`Validation examples <examples_valid>`

   .. grid-item-card:: Physics Simulation

      Use Rabbit FEM and Mooseherder to run simulations, generate meshes,
      orchestrate parameter sweeps and fit simulation parameters.

      +++

      :doc:`Simulation examples <examples_mooseherder>`

   .. grid-item-card:: Data I/O

      Load experimental sensor traces and probabilistic simulation samples.

      +++

      :doc:`Data I/O examples <examples_dataio>`

.. toctree::
   :hidden:
   :maxdepth: 2

   examples_basics_sensorsim
   examples_dic
   examples_render3d
   examples_renderuvs
   examples_ext_sensorsim
   examples_valid
   examples_mooseherder
   examples_dataio
