.. image:: /_static/pyvale_logo.png
   :alt: PyVale
   :width: 420px
   :align: center

Your virtual engineering laboratory
====================================

Design experiments, analyse measurements and iterate with evidence.
Experiments are expensive make every measurement count.

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install pyvale

   .. tab-item:: uv

      .. code-block:: bash

         uv add pyvale

Python 3.11+.
:doc:`Installation guide </install/install>`

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: SensorSim

      .. image:: /_static/basics_ex0_locs.png
         :alt: Virtual sensor locations
         :height: 145px
         :align: center

      Build virtual sensor arrays with realistic uncertainty. Run repeated experiments
      and inspect measurement distributions.

      +++

      :doc:`First example </examples/basicsensorsim/ex0_quickstart>` · :doc:`User guide </guide_user/guide_sensorsim>`

   .. grid-item-card:: Digital Image Correlation

      .. image:: /_static/dic_ex10_3d.png
         :alt: Stereo DIC reconstructed surface
         :height: 145px
         :align: center

      Analyse 2D and stereo images to recover shape, displacement and strain from
      synthetic or experimental images.

      +++

      :doc:`First example </examples/dic/ex02_plate_with_hole>` · :doc:`User guide </guide_user/guide_dic>`

   .. grid-item-card:: Render

      .. image:: /_static/render3d_ex1d_riley_dicuq.png
         :alt: Synthetic camera images generated with Riley
         :height: 145px
         :align: center

      Render deforming meshes with Riley, integrated into PyVale. Configure cameras,
      textures, distortion and point spread functions; use optional Blender for path
      tracing.

      +++

      :doc:`First example </examples/render3d/ex1a_riley_quickstart>` · :doc:`Examples </examples/examples_render3d>`

   .. grid-item-card:: Physics Simulation

      .. image:: /_static/physics_simulation_field.png
         :alt: Raw Exodus temperature field on the undeformed finite element mesh; no sensors
         :height: 145px
         :align: center

      Use Rabbit FEM through PyVale’s Mooseherder module to run thermo-mechanical finite
      element simulations, vary inputs and load results for virtual experiments.

      +++

      :doc:`First example </examples/mooseherder/ex01a_quickstart_rabbitfem>` · :doc:`User guide </guide_user/guide_mooseherder>`

SensorSim, DIC and Render are ready for general use. In development: sensor
placement optimisation, simulation calibration and validation, and a workflow
engine that brings PyVale’s modules together for reproducible virtual
experimental campaigns, from individual studies to large-scale sweeps.

.. grid:: 2
   :gutter: 3

   .. grid-item::

      .. image:: /_static/riley_logo.png
         :alt: Riley — rendering backend integrated with PyVale Render
         :height: 155px
         :align: center
         :target: https://github.com/Computer-Aided-Validation-Laboratory/riley-raster

      .. centered:: Riley · PyVale Render

   .. grid-item::

      .. image:: /_static/logo_rabbitfem.png
         :alt: Rabbit FEM — simulation backend used through Mooseherder
         :height: 155px
         :align: center
         :target: https://github.com/Computer-Aided-Validation-Laboratory/rabbit-fem

      .. centered:: Rabbit FEM · Mooseherder

----

Developed by the Computer Aided Validation Team and collaborators.


.. toctree::
    :hidden:

    install/install
    examples/examples
    guide_user/guide_user
    guide_theory/guide_theory
    guide_dev/dev_guide
    api_py
    api_cpp
    cite
