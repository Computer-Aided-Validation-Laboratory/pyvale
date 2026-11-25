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
         :target: examples/basicsensorsim/ex0_quickstart.html

      Build virtual sensor arrays with realistic uncertainty. Run repeated experiments
      and inspect measurement distributions.

      +++

      :doc:`First example </examples/basicsensorsim/ex0_quickstart>` · :doc:`User guide </guide_user/guide_sensorsim>`

   .. grid-item-card:: Digital Image Correlation

      .. image:: /_static/dic_ex10_3d.png
         :alt: Stereo DIC reconstructed surface
         :height: 145px
         :align: center
         :target: examples/dic/ex01_region_of_interest.html

      Analyse 2D and stereo images to recover shape, displacement and strain from
      synthetic or experimental images.

      +++

      :doc:`First example </examples/dic/ex01_region_of_interest>` · :doc:`User guide </guide_user/guide_dic>`

   .. grid-item-card:: Render

      .. image:: /_static/render3d_ex1d_riley_dicuq.png
         :alt: Synthetic camera images generated with Riley
         :height: 145px
         :align: center
         :target: examples/render3d/ex1a_riley_quickstart.html

      Render deforming meshes with Riley, integrated into PyVale. Configure cameras,
      textures, distortion and point spread functions; use optional Blender for path
      tracing.

      +++

      :doc:`First example </examples/render3d/ex1a_riley_quickstart>` · :doc:`Examples </examples/examples_render3d>`

   .. grid-item-card:: Validation

      .. image:: /_static/val_ex10_fig03_mavm_support_envelope_TC3.png
         :alt: MAVM support envelope for thermocouple TC3
         :height: 145px
         :align: center
         :target: examples/validation/ex00_valid_quickstart.html

      Compare simulation predictions with experimental measurements and quantify
      discrepancies under uncertainty.

      +++

      :doc:`First example </examples/validation/ex00_valid_quickstart>` · :doc:`Examples </examples/examples_valid>`

   .. grid-item-card:: Mooseherder

      .. image:: /_static/physics_simulation_field.png
         :alt: Raw Exodus temperature field on the undeformed finite element mesh; no sensors
         :height: 145px
         :align: center
         :target: examples/mooseherder/ex01a_quickstart_rabbitfem.html

      Use Rabbit FEM through PyVale’s Mooseherder module to run thermo-mechanical finite
      element simulations, vary inputs and load results for virtual experiments.

      +++

      :doc:`First example </examples/mooseherder/ex01a_quickstart_rabbitfem>` · :doc:`User guide </guide_user/guide_mooseherder>`

   .. grid-item-card:: DataIO

      .. image:: /_static/dataio_ex01_pulse253_tc3.png
         :alt: Pulse253 CSV excerpt alongside the loaded TC3 temperature trace
         :height: 145px
         :align: center
         :target: examples/dataio/ex01_load_exp_data.html

      Load experimental sensor measurements and probabilistic simulation data for
      analysis in PyVale.

      +++

      :doc:`First example </examples/dataio/ex01_load_exp_data>` · :doc:`Examples </examples/examples_dataio>`

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
    api_py
    api_cpp
    cite
