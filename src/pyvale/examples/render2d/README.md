# 2D image-warp examples

These examples use the planar `IImageWarp2D` interface. The PixInt2D
renderers (`ex1_*`) provide analytic Grid2D and Speck2D images through
Riley-ordered Newton element mappings.

The `ImageDef2D` example (`ex2_*`) warps a reference image from finite-element
displacements and is provided for legacy compatibility only: prefer Riley
even for 2D rendering as it is faster and more deeply verified for
correctness.
