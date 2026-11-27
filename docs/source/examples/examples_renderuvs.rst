.. _examples_renderuvs:

Render UVs
==========

Control how textures map onto mesh surfaces. Start with planar projection and
texture aspect ratios, compare boundary behaviour, then project onto arbitrary
planes and transform the mapping. These tools support the scenes demonstrated
in :doc:`examples_render3d`.

Planar projection and texture fit
---------------------------------

Choose projection axes and preserve or adjust the texture aspect ratio.

.. toctree::
   :maxdepth: 1

   renderuvs/ex1a_uv_planar_axes
   renderuvs/ex1b_uv_texture_aspect

Texture boundaries
------------------

Compare how sampling behaves outside the texture region.

.. toctree::
   :maxdepth: 1

   renderuvs/ex1c_uv_pixel_region

Arbitrary planes and transforms
-------------------------------

Project onto an arbitrary plane, then translate, rotate and scale the mapping.

.. toctree::
   :maxdepth: 1

   renderuvs/ex1d_uv_arbitrary_plane
   renderuvs/ex1e_uv_transform
