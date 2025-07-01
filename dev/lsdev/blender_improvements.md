# Improvements to be made to pyvale's Blender capability

Outputting the rendered image as an array more efficiently
- See if it's possible to save image render array without saving to disk first

Mapping speckle patterns to non-flat geometry
- UV unwrap tool in pyvale currently only caters a cube uv unwrap.
- This works for flat geomtry but won't for curved or non-flat geometry

Improve lighting capability
- Blender supports multiple different light types, each with their own parameters that can be set
- pyvale only allows the user to set a limited number of parameters, and there is not currently the option to set a specific parameter given a choice of light type
- It would also be nice to have presets of different lights, so that the user doesn't have to specify every parameter each time
- Also a bit unsure how it uses/calculates the `energy` for the lights

Option for rendering on the GPU
- Currently it is set to only run on the CPU, not the GPU.
- Would need to make sure the way Blender is coded is compatible to the GPU available

Rendering multiple deforming objects
- Currently the code only works to image a single deforming object, and it centres this object
- It should be relatively easy to add other deforming objects, but this centring would need to be changed
- Not sure how the shape keys would need to be adapted - might need multiple linked shape keys(?)

Bug in CameraTools stereo methods
- When I deep copy the CameraData dataclass, some of the parameters don't update to match new camera position.
