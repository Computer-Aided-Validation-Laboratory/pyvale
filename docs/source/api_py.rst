Detailed Python API
===================

Choose an API by the task you want to perform. The module pages below contain
reference details; the :doc:`user guides <guide_user/guide_user>` and
:doc:`examples <examples/examples>` explain the workflows.

Load simulation and experimental data
------------------------------------

Use ``pyvale.dataio`` for ``SimData``, experimental data containers, loaders,
mesh conventions and conversions. ``pyvale.data`` retrieves demonstration data
and simulation inputs. See :doc:`Working with SimData <guide_user/guide_simdata>`.

.. toctree::
   :maxdepth: 1

   pyvale.dataio
   pyvale.data

Simulate sensor measurements
----------------------------

Build sensor arrays, error chains and repeated virtual experiments.

.. toctree::
   :maxdepth: 1

   pyvale.sensorsim

Generate camera images
----------------------

Create scenes and synthetic images through the Render API. Blender's supported
backend is documented within ``pyvale.render`` alongside the other rendering
tools. See :doc:`the Render guide <guide_user/guide_render>`.

.. toctree::
   :maxdepth: 1

   pyvale.render

Analyse images and calibrate cameras
-----------------------------------

Use ``pyvale.dic`` for correlation, ``pyvale.calib`` for camera calibration and
``pyvale.strain`` for strain calculation. For a worked calibration, start with
:doc:`the stereo calibration example <examples/dic/ex07_calibration>`.

.. toctree::
   :maxdepth: 1

   pyvale.dic
   pyvale.calib
   pyvale.strain

Run physics simulations and sweeps
---------------------------------

Launch meshing and solver runs, organise parameter studies and load their results.

.. toctree::
   :maxdepth: 1

   pyvale.mooseherder

Supporting verification tools
-----------------------------

``pyvale.verif`` provides analytic fields and mesh generators for checking
numerical behaviour. These helpers are distinct from simulation-validation
metrics and calibration workflows, which are under development. Unfinished
validation packages and internal backend/native implementation modules are
outside this public reference.

.. toctree::
   :maxdepth: 1

   pyvale.verif
