.. _guide_overview:

User Guide
=================================

Module user guides
------------------

.. toctree::
   :maxdepth: 1

   guide_sensorsim.rst
   guide_dic.rst
   guide_blender.rst


What is pyvale?
---------------------------------

Pyvale is a virtual engineering laboratory that allows you to simulate sensor deployments, optimise experimental design and calibrate/validate simulations. The motivation behind pyvale is to provide engineers with a software tool that they can import a physics simulation and virtually design a simulation validation experiment without needing to go to the laboratory. 

Pyvale will never completely replace the need to obtain real experimental data to validate engineering physics simulations. What it can do is reduce the time and cost of experiments by allowing you to perform smarter and more resilient experiments - while providing you the tools to analyse your experimental data to calibrate your simulation or calculate validation metrics.

Another key motivation for pyvale was to provide open-source tools for simulating imaging sensors (e.g. digital image correlation and infra-red thermography) that could be used for large parallel design sweeps on computing clusters without licensing restrictions or the need to build from source. With pyvale we aim to make it easy for experimentalists and simulation engineers to simulate their imaging experiments with a simple python interface and underlying performant code.
Pyvale consists of three overarching toolboxes which can be used for:
 
#. Sensor uncertainty quantification simulation. 
#. Experimental design and sensor placement optimisation.
#. Simulation calibration and validation metrics. 

Current progress on pyvale has built the foundation of the sensor simulation engine as this will be used a virtual test bed for the other toolboxes. The pyvale sensor simulation engine can produce visualisations of virtual sensor deployments and simulated sensors traces with realistic uncertainties as shown below.  

.. |image1| image:: ../_static/thermomech3d_tc_vis.png
   :width: 49%
   :align: middle

.. |image2| image:: ../_static/thermomech3d_tc_traces.png
   :width: 49%
   :align: middle

|image1| |image2|


What goes on under the hood?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The core sensor simulation engine in pyvale leverages a suite of scientific computing packages in python including: numpy, scipy, matplotlib and pyvista. All computations that are performed by pyvale in python are implemented as vectorised operations on numpy arrays to improve performance. The results from pyvale sensor simulations are returned as numpy arrays allowing for interoperability with tools like matplotlib for visualisation and scipy for statistical analysis.

For the computationally expensive modules in pyvale such as the simulation of imaging sensors and processing of imaging data (e.g. digital image correlation) we use compiled code that only has minimal interaction with the python interpreter. This means that the user can setup and run a ray-tracing simulation for digital image correlation and then perform digital image correlation on the generated images with only a few lines of python code while maintaining good performance.


How do I get started with pyvale?
---------------------------------

You can install pyvale from PyPI using ``pip install pyvale``. We recommend a virtual environment with python 3.11. If you are a new python user and need help setting up the correct python version and your virtual environment then go to our detailed :ref:`install guide <install_all>`.

Once you have pyvale installed you should get familiar with the core concepts of pyvale starting with the :ref:`basic examples <examples_sensorsim_basics>`. These examples come with some pre-packaged simulation data so you will not need to provide your own simulation to get started. With the core functionality of pyvale you will have everything you need to be able to build any custom sensor array that samples scalar (e.g. temperature), vector (e.g. displacement, velocity) or tensor fields (e.g. strain).

After that you might want to look at some of the camera sensor simulation tools in pyvale including our digital image correlation rendering module (based on Blender) and our digital image correlation processing module.


Why is there a rabbit on our logo?
-----------------------------------

We like rabbits and we like digital image correlation - so a rabbit with a speckle pattern seemed like a good idea. Do we need another reason?

.. image:: ../_static/pyvale_rabbit_only.png
   :alt: pyvale rabbit only
   :align: center
   :width: 400px

We could have said it was some forced metaphor that rabbits multiply a lot and one of the main applications of pyvale is running large parallel sweeps on computing clusters. But it was actually just a series of jokes from different team members. Someone said "I can't design the logo because if I do it will have a rabbit on it", then someone else said that "A rabbits ears look like the 'V' in pyvale". And that was how our logo was born.


