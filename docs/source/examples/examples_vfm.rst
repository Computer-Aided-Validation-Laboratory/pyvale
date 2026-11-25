.. _examples_vfm:

Virtual Fields Method
=====================

Identify elastoplastic material properties from full-field strain and force
data. Each example generates its tensile experiment in memory.

Getting started
---------------

Identify uniform material properties using sensitivity-based virtual fields.

.. toctree::
   :maxdepth: 1

   vfm/vfm_ex1_hom

Spatially varying properties
----------------------------

Estimate yield strength by slice, refine the slice boundaries, or combine
homogeneous and slice-wise identification in a single workflow.

.. toctree::
   :maxdepth: 1

   vfm/vfm_ex2_slicewise
   vfm/vfm_ex3_slicewise_refinement
   vfm/vfm_ex4_hom_slicewise
