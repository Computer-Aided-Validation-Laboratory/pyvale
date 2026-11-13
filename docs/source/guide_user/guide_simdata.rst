.. _guide_simdata:

Working with SimData
========================

Bring simulation results into SensorSim or Render without tying your analysis
to a particular solver. ``pyvale.dataio.SimData`` groups reference geometry,
connectivity, times and physical fields as NumPy arrays.

Before you start
--------------------

Identify the units, coordinate system, field names and node ordering in your
source. ``SimData`` does not infer or convert their physical meaning. Attributes
may be ``None`` when absent or not selected. A temperature array is a scalar
field; displacement components are separate named scalar arrays. The container
does not infer vector meaning from field names: select the component names and
coordinate directions explicitly for downstream sensors or renderers.

1. Load only what you need
------------------------------

Use the Exodus loader as a context manager. ``get_read_config()`` lists
available fields. A fresh ``SimLoadConfig`` includes geometry/time by default,
but variable selections must be supplied explicitly.

.. code-block:: python

   import pyvale.data as dataset
   import pyvale.dataio as io
   import pyvale.mooseherder as mh

   exodus_file = dataset.thermomechanical_2d_path()
   with mh.ExodusLoader(exodus_file) as loader:
       available = loader.get_read_config()
       print(available)
       simulation = loader.load_sim_data(io.SimLoadConfig(
           node_vars=("temperature",),
           time_inds=slice(-1, None),
       ))

   # Arrays remain valid after the NetCDF file is closed.
   print(simulation.coords.shape)  # nodes, three coordinate columns
   print(simulation.time.shape)    # one retained sample
   print(simulation.node_vars["temperature"].shape)  # nodes, one time
   final_temperature = simulation.node_vars["temperature"][:, 0]
   print(final_temperature.min(), final_temperature.max())


This selects the last saved time while preserving a length-one time axis. The
loaded arrays remain usable after the file closes.

2. Understand the array axes
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Attribute
     - Shape or structure
     - Meaning
   * - ``coords``
     - ``(nodes, 3)``
     - Reference coordinates; columns are x, y, z.
   * - ``time``
     - ``(times,)``
     - Saved simulation times in source units.
   * - ``connect``
     - Block key → ``(elements, nodes_per_element)``
     - Connectivity for each mesh block after default loader normalisation.
   * - ``node_vars``
     - Field name → ``(nodes, times)``
     - Each scalar component has its own name.
   * - ``elem_vars``
     - ``(field_name, block_position)`` → ``(elements, times)``
     - Block positions in ExodusLoader selections are one-based.
   * - ``glob_vars``
     - Name → ``(times,)``
     - Global results such as a reaction force.

Default ``ExodusLoader.load_sim_data()`` normalises connectivity to element
rows and zero-based node indices. Its low-level ``get_connectivity()`` accessor
does not. Exodus side-set node/element numbers remain one-based; subtract one
from node numbers before indexing a NumPy coordinate or field array. Side-set
element numbers alone are not complete surface connectivity.

3. Use your own result files
--------------------------------

Construct ``SimData`` directly from arrays, or use ``SimLoaderByField`` /
``SimLoaderByTime`` for text and NumPy exports. Keep every field's node order
aligned with ``coords`` and every time column aligned with ``time``. For a
planar model, coordinates still have three columns; ``num_spat_dims`` records
the active dimension. Preserve block identity, element type and the local node
convention when supplying connectivity. These describe how a field is
interpolated, not just how arrays are stored.

Matching shapes do not prove correspondence: node rows must refer to the same
physical points, and time columns to the same saved times. Keep that mapping
explicit when combining exports from another solver.

Analyse the result
------------------

Compare coordinate bounds, saved times and a known field value with the source
simulation. Interpret a selected sample in its original unit system before
calculating derived quantities. Make a copy before modifying a slice: NumPy
views can share storage. Preserve raw fields while preparing converted units,
filtered fields or other derived arrays.

SensorSim samples the selected fields. Render converts the geometry to visible
surface meshes using an explicit connectivity convention and named displacement
components. Its displacement axes differ from a SimData nodal field:
``Mesh3D.displacements`` is ``(frames, nodes, 3)``. Verify that correspondence
when handing motion to a renderer.

Unit changes must include all affected coordinates, displacements and camera
lengths. For multiple cases, preserve case-to-parameter identity as well as
array axes so that comparisons represent the intended simulation choices.

Continue with these examples and guides:

:doc:`SimData inspection example </examples/mooseherder/ex01b_quickstart_simdata>` · :doc:`Custom data examples </examples/examples_ext_sensorsim>` · :doc:`Data I/O API </pyvale.dataio>`

:doc:`Generate images <guide_render>` · :doc:`Run physics cases <guide_mooseherder>`
