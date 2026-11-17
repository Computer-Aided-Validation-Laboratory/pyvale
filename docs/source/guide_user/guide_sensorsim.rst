.. _guide_sensorsim:

SensorSim User Guide
====================

SensorSim predicts what a sensor array would measure from a physics simulation.
Use it to explore sensor locations, sampling and uncertainty before an
experiment. The same workflow supports scalar fields such as temperature,
vector fields such as displacement, and tensor fields such as strain.

The workflow
------------

1. **Load a simulation.** Read Exodus results with Mooseherder, or load your own
   field arrays into ``pyvale.dataio.SimData``. Keep coordinates, field values
   and sampling times in consistent units.
2. **Define the sensors.** Choose the field components, sensor positions,
   sampling times and, where applicable, orientations and spatial averaging.
3. **Attach uncertainty models.** Build an ordered error chain describing the
   effects relevant to your measurement.
4. **Repeat and inspect.** Run virtual experiments, calculate statistics and
   visualise the simulated measurements.

:doc:`Start with the SensorSim quickstart </examples/basicsensorsim/ex0_quickstart>`.

Measurements and uncertainty
----------------------------

The simulation supplies the reference field used as “truth” within the virtual
experiment. It is a modelling assumption, not a claim that the physics model
matches reality. SensorSim samples that field at the sensor locations and times.

.. math::

   \mathrm{measurement} = \mathrm{truth}
       + \mathrm{systematic\ error} + \mathrm{random\ error}

An error chain can represent effects such as bias, noise, digitisation,
saturation and uncertainty in position, orientation or sampling time. Systematic
and random describe the simulated error behaviour; Type A and Type B refer to
how uncertainty is evaluated and are not synonyms for those categories.

The order matters. An independent error uses the truth as its reference; a
dependent error uses the accumulated measurement at that point in the chain.
For example, percentage noise may be relative to the truth or to a measurement
already affected by earlier errors.

Repeated virtual experiments
----------------------------

Apply one or more sensor arrays to one or more simulation cases, with repeated
error realisations. Use ``ExperimentSimulator`` to generate measurement and
error arrays, then compare their distributions or summary statistics. Parallel
execution can be configured for larger studies.

A measurement array has shape ``(sensors, components, times)``. Repeated
experiment arrays add a leading axis:
``(experiments, sensors, components, times)``. Statistics over the first axis
summarise variation between virtual experiments; they do not establish that the
chosen uncertainty models capture every effect in a real measurement.

Bring your own data
-------------------

Use ``SimData`` to describe coordinates, field arrays and times, together with
connectivity for mesh-based simulations. PyVale provides loaders for Exodus,
delimited text and NumPy arrays. Mesh topology and node ordering must follow the
supported conventions; units and coordinate systems must be consistent with
your sensor definitions.

:doc:`Custom data and uncertainty examples </examples/examples_ext_sensorsim>` · :doc:`Data I/O API </pyvale.dataio>`

Where to go next
----------------

:doc:`Basic examples </examples/examples_basics_sensorsim>` · :doc:`Extended examples </examples/examples_ext_sensorsim>` · :doc:`SensorSim API </pyvale.sensorsim>`
