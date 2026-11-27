Detailed C++ API
================

This reference covers the native DIC and strain implementations behind PyVale's
Python interfaces. Application workflows are documented separately in the
:doc:`Python API <api_py>`.

Native call boundaries
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Entry point
     - Buffer layout and dtype
     - Ownership and output
   * - ``engine`` / ``engine_images``
     - The ROI is a C-contiguous bool array, ``(height, width)``. Native images
       contain row-major uint8, uint16, uint32 or float32 pixels.
     - ROI and supplied Image objects are borrowed for the call. The Python
       image-array binding copies pixels into owned native buffers. Results
       are written to the configured files.
   * - ``strain::engine_2d`` / ``strain::engine_3d``
     - Subset positions are C-contiguous int32 grids, ``(rows, columns)``.
       Field arrays are float64, ``(frames, rows, columns)``, in C order.
     - Inputs remain alive for the synchronous call. Internal windows and
       result buffers are owned by the engine; output is written to files.

See :doc:`the DIC entry points <cpp/dic/dicmain>` and
:doc:`the strain entry points <cpp/strain/strain>` for the signatures and buffer
contracts. Native code reads these layouts directly, so arbitrary strided NumPy
views must be normalised at the Python boundary before the call.

Units and coordinate conventions
--------------------------------

DIC subset positions and 2D displacements are in image pixels. Stereo
reconstruction uses the calibration's length unit, currently millimetres in
PyVale's calibration API. Two-dimensional strain fits use pixel-grid positions
and displacements; three-dimensional surface fits use physical positions and
physical displacements in the same length unit. Keep frame, row and column axes
consistent when preparing native inputs.

Implementation headers
----------------------

The generated pages below describe the current headers. Start with the entry
points above when changing a Python/native boundary; individual interpolation,
matching and fitting helpers are implementation details.

.. toctree::
   :maxdepth: 1

   cpp/cpp_dic
   cpp/cpp_strain
