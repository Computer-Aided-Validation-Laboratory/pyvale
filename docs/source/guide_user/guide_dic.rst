

.. _guide_dic:

DIC User Guide
==============

This page should be conisered more of a guide to the key interactions and useful
tips and tricks for getting good results in a reasonable amount of time.
A high level overview of the Pyvale '*workflow*' Can be found in the flowchart below.


The Pyvale workflow
------------------


Importing the DIC modules
^^^^^^^^^^^^^^^^^^^^^^^^^^

After installing Pyvale, you'll need to importing the relevent modules before
going any further. We like to import the modules in the following way:

.. code-block:: Python

    import pyvale.dic as dic
    import pyvale.strain as strain



.. figure:: guide_dic_flowchart.png
    :alt: DIC flowchart
    :width: 60%

For any DIC calculation the user must first specify the **region of interest
(ROI)** for their calculation. You can either use Pyvale for this (see an
in-depth example :doc:`here <../examples/dic/ex1_region_of_interest.rst>`), 
or, you can create it using Numpy. Simple examples can be found below:

.. tab-set::

   .. tab-item:: Interactive ROI Pyvale

      .. code-block:: python

         roi = dic.RegionOfInterest(ref_image="./ref_img.tiff")
         roi.interactive_selection(subset_size=31)  # ROI GUI will launch

         dic.calculate_2d(
             ...,
             roi=roi.mask,
             seed=roi.seed,
             ...,
         )

   .. tab-item:: Programmatic ROI Pyvale

      .. code-block:: Python

         roi = dic.RegionOfInterest(ref_image="./ref_img.tiff") # set reference image
         roi.rect_boundary(left=100,right=100,bottom=100,top=100) # exclude a 100 pixel boundary

         dic.calculate_2d(
             ...,
             roi=roi.mask,
             seed=[500, 500],  # seed at centre of image
             ...,
         )

   .. tab-item:: Programmatic ROI Numpy

      .. code-block:: Python

         arr = np.zeros((1000, 1000), dtype=bool)  # image is 1000x1000
         arr[100:900, 100:900] = True  # Exclude a 100 pixel boundary

         dic.calculate_2d(
             ...,
             roi=arr,
             seed=[500, 500],  # seed at centre of image
             ...,
         )

Performing a correlation
^^^^^^^^^^^^^^^^^^^^^^^^

The next step is to perform a correlation with the dic.calculate_2d function.
There are a few arguments that **must** be passed. These are the
rference and deformed images, the roi mask and seed, as well as the subset size
and subset step:

.. code-block:: Python

   dic.calculate_2d(
        reference=ref_image, # Can be str | pathlib.Path | np.ndarray
        deformed=def_image,  # Can be str | pathlib.Path | np.ndarray
        roi_mask=roi.mask,   # np.ndarray
        seed=roi.seed,       # pair of intergers. Allowed as list[int] | list[np.int32] | np.ndarray 
        subset_size=31,      # must be an odd number
        subset_step=15
   )

In the simplest above case, all other arguments will have default values. See
the API documentation for this function for more details.

Understanding Output files
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The next step is to understand the output. By default the results will be saved
in the users current working directory in human readable .CSV format with a filename prefix of
:code:`dic_results_` followed by the name of the deformed image. The output will have
the following columns:

- **subset_x**:  
  X-coordinate of the center of the subset (or window) used in displacement tracking or correlation analysis.

- **subset_y**:  
  Y-coordinate of the center of the subset used in displacement tracking or correlation analysis.

- **displacement_u**:  
  Displacement in the X-direction (horizontal) calculated for the subset.

- **displacement_v**:  
  Displacement in the Y-direction (vertical) calculated for the subset.

- **displacement_mag**:  
  Magnitude of the displacement vector.

- **converged**:  
  Boolean flag indicating whether the displacement calculation algorithm converged for this subset.

- **cost**:  
  The final value of the cost function used during the displacement calculation.
  The reported value is always given as the **ZNCC** no matter if the SSD, NSSD or ZNSSD has been chosen as the correlation function.
  The ZNCC is calculated with the final parameter values from the last optimizer iteration. 

- **ftol**:  
  The final value of the function tolerance, a measure of how much the cost function changed between iterations at convergence.

- **xtol**:  
  The final value of the solution tolerance, a measure of how much the solution (displacement) changed between iterations at convergence.

- **num_iterations**:  
  The number of iterations the algorithm took to converge for this subset.

You can alter the path, delimiter and filename prefix of the output file using the arguments
:code:`output_basepath`, :code:`output_delimiter` and :code:`output_prefix`. You
can also opt to save results in binary format. This can be done by setting
:code:`output_binary=True`.

Importing DIC Results
^^^^^^^^^^^^^^^^^^^^^

Once you have finished your correlation, you can proceed with whatever
visualization and post-processing tools/software you'd like. Alternatively,
Pyvale provides the option to read the data in using a single command into a
single dataclass that can be used for easy plotting. Importing data is done with
the :code:`dic.import_2d`` command. The below highlights how to import data and
create a simple plot of the displacement

.. :code-block:: Python

   import matplotlib.pyplot as plt

   dic_data = dic.import_2d(data="./dic_results_*")

   # plot of vertical displacement for first deformation image.
   plt.pcolor(dic_data.ss_x, 
           dic_data.ss_y, 
           dic_data.u_y[0]) # [image, y, x]

The import will find all files in the current working directory with that
filname prefix. If you have changed :code:`output_delimiter` prior to the
correlation you will also need to specify the delimiter when importing the data.

Strain Calculation
^^^^^^^^^^^^^^^^^^^

The previous step is optional if you are wanting to perform a strain
calculation. If you don't need to do any kind of visualization or analysis, you
can import DIC data and calculate strains using 


.. tab-set::

   .. tab-item:: Embedded DIC data Import

      .. code-block:: python

         strain.calculate_2d(data="./dic_results_*",
            input_delimiter=",",
            window_size=5,
            window_element=9
         )

   .. tab-item:: Prior DIC data Import

      .. code-block:: Python

         dic_data = dic.import_2d(data="./dic_results_*", delimiter=",")

         strain.calculate_2d(
            data=dic_data,
            window_size=5,
            window_element=9
         )


Importing DIC Data
^^^^^^^^^^^^^^^^^^^^



DIC Methods
------------



Multiwindow RG-DIC
^^^^^^^^^^^^^^^^^^^

Incremental RG-DIC
^^^^^^^^^^^^^^^^^^^

Image Scan
^^^^^^^^^^^^

While not generally recommended for obtaining accurate results. This is useful 


DIC with Large Images/Displacements
-------------------------------

Setting a Maximum Displacement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically if you are performing DIC with large displacements that will exceed
well above 100 pixels, then there will be a few arguments that you will want to
consider tweaking that might help to improve your results.

Firstly, it's important to chose a max displacement value that is comfortably
larger than your estimate for the final maximum displacement. For example, if
you *think* your max displacement is roughly 300 pixels, then trying a value of
512 (powers of two help with FFT efficiency) would be a sensible option. This
can be done with:

.. code-block:: Python
   :emphasize-lines: 3

   dic.calculate_2d(
        ...,
        max_displacement=512,
        ...,
   )

Pyvale will always round the `max_displacement` value up to the next greatest
power of 2. So if you select 400, it would still round to 512. Again, this is to
benefit from the efficiencies of FFTs.

Enabling Outlier Removal in Multiwindow FFTCC Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The FFTCC multiwindowing approach involves seeding smaller windows with the
rigid estimates from neighbouring points in the previous larger window. This can
sometimes be problematic when multiple windows are involved as incorrect
estimates can be propogated through the smaller windows, leading to a wildly
incorrect initial rigid estimate for the displacements.

Pyvale has a Median Absolute Deviation (MAD) outlier removal
flag that, when enabled, will kill likely incorrect spikes in the rigid
estimates or each FFTCC window size. This can be enabled with the following
arguments when calling the DIC engine:

.. code-block:: Python
   :emphasize-lines: 3-4

   dic.calculate_2d(
        ...,
        fft_mad=True,
        fft_mad_scale=3.0, # <-- Default value
        ...
   )

The MAD outlier removal works in the following way:

#. Looks at nearby subsets in a 2D neighborhood
#. Computes the median of their shifts
#. Computes the MAD (median absolute deviation)
#. If the current value deviates **too much** from the neighborhood,
   it replaces it with the local median.

**too much** is defined by the `mad_scale` argument. A value is deemed as
replaceable only if :math:`| x − \mathrm{median}| > \mathrm{fft\_mad\_scale} \times \mathrm{MAD}`, 
A larger `fft_mad_scale` is therefore more *tolerant*, while a smaller value kills larger deviations.


Sequential Image Loading
^^^^^^^^^^^^^^^^^^^^^^^^

When working with a series of large images, RAM usage starts to become an
important consideration. In it's current form, Pyvale will read **all** images
in the workflow when it starts. This isn't a huge problem for typical DIC
workflows where images are typically 10s of MBs, but will start to cause crashes
with high resolution images (100s MBs.). To get around this we'd recommend
**placing the DIC engine call in a loop over the images**. An example of which can be found
below:

.. code-block:: Python

    ref_img = "ref_00.tiff"
    def_imgs = ["def_00.tiff", "def_01.tiff", "def_02.tiff", ...]

    for def_img in def_imgs:
        dic.calculate_2d(reference=ref_img,
                         deformed=def_img,
                         ...)

There are plans to change this in later pyvale versions so that images
are read sequentially and thus avoiding the need for any loops. Please keep an
eye on the documentation for any future changes.


Selecting a Thread Count
-------------------------------

Understanding DIC Output
-------------------------------

Incremental DIC
-----------------

This feature is a Work in Progress..
