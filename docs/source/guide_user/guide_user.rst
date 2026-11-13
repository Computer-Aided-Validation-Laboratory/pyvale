.. _guide_overview_user:

User Guide
==========

Experiments are expensive. Before committing time, specimens and laboratory
resources, you need to know what to measure and whether those measurements can
answer your engineering question. PyVale lets you explore measurement choices
and uncertainty using a physics simulation, then analyse the measurements you
collect.

The aim is to connect predictions with experimental evidence for simulation
validation. A virtual experiment helps you investigate what sensors and cameras
would observe under your assumptions; real measurements provide the evidence
needed to assess those assumptions.

Start with what you have
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Your starting point
     - Workflow
     - What you obtain
   * - Simulation fields and a sensor plan
     - **SensorSim**
     - Virtual measurements and uncertainty distributions.

   * - Camera images
     - **Digital Image Correlation**
     - Shape, displacement and strain.

   * - A mesh and camera model
     - **Render**
     - Synthetic camera images with controlled imaging conditions.

   * - Solver inputs and parameter choices
     - **Physics Simulation**
     - Physics fields and results from a collection of simulation cases.

Follow the relevant route:

* **Simulate measurements:** :doc:`SensorSim guide <guide_sensorsim>` · :doc:`First example </examples/basicsensorsim/ex0_quickstart>`
* **Analyse images:** :doc:`DIC guide </guide_user/guide_dic>` · :doc:`Examples </examples/examples_dic>`
* **Generate camera images:** :doc:`Render examples </examples/examples_render3d>` · :doc:`UV mapping </examples/examples_renderuvs>`
* **Run solver sweeps:** :doc:`Mooseherder guide </guide_user/guide_mooseherder>` · :doc:`First simulation </examples/mooseherder/ex01a_quickstart_rabbitfem>`

Connect predictions and measurements
------------------------------------

A physics simulation predicts fields. SensorSim predicts what a proposed sensor
array would measure from those fields. Render turns a deforming mesh into
camera images, and DIC turns images into deformation measurements. Experimental
images can enter the DIC workflow directly; a physics solve is not a prerequisite
for analysing them.

These connections help you design measurements that can discriminate between
predictions and investigate uncertainty before collecting validation data.

SensorSim, DIC and Render are available for measurement simulation and image
analysis. Rabbit FEM and Mooseherder support physics simulations and parameter
sweeps. Dedicated tools for sensor placement optimisation, simulation
calibration and validation, and a workflow engine for reproducible experimental
campaigns are under development. Today, you connect the modules in Python.

Getting started
---------------

Use Python 3.11 or newer for core PyVale workflows. Riley is the default
starting point for rendering. The optional Blender backend requires Python
3.13 and ``pip install "pyvale[blender]"``.

:doc:`Installation guide </install/install>` · :doc:`Browse examples </examples/examples>`

Example data is downloaded on demand and cached locally. If you already have
simulation results or camera images, start directly with the relevant workflow.

Why is there a rabbit on our logo?
----------------------------------

We like rabbits and we like digital image correlation, so a rabbit with a
speckle pattern seemed like a good idea.

.. image:: /_static/pyvale_rabbit_only.png
   :alt: PyVale's rabbit with a DIC speckle pattern
   :width: 220px
   :align: center

The design grew out of a few team jokes: “I can’t design the logo because if I
do it will have a rabbit on it,” followed by, “A rabbit’s ears look like the V
in PyVale.” We could claim a metaphor about rabbits multiplying like parallel
simulation runs, but the rabbit came first.

.. toctree::
   :hidden:
   :maxdepth: 1

   guide_sensorsim
   guide_dic
   guide_blender
   guide_strain
   guide_mooseherder
