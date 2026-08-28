# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Unified rendering APIs for pyvale.

The namespace separates validated three-dimensional scene renderers from
validated planar image-warp renderers. The default 3D implementation is
:class:`Riley`; :class:`ImageDef2D` provides the initial 2D implementation.
"""

from .blender import (
    Blender,
    BlenderCalibrationData,
    BlenderCalibrationTarget,
    BlenderConfig,
    BlenderImageShader,
    BlenderMaterial,
    BlenderTextureShader,
    EBlenderDevice,
    EBlenderEngine,
    blender_available,
    blender_camera_from_resolution,
    blender_field_of_view,
    blender_gpu_available,
    blender_mm_per_pixel,
    calibration_image_count,
    focal_length_from_resolution,
    raise_if_blender_unavailable,
    render_calibration_images,
)
from .camera import Camera, Camera2D, EDistortionModel, EPSFType
from .camerastereo import CameraStereo
from .cameratools import (
    average_subpixel_image,
    crop_image_rectangle,
    faceon_stereo_cameras,
    pixel_grid_leng,
    pixel_vec_leng,
    subpixel_grid_leng,
    subpixel_vec_leng,
    symmetric_stereo_cameras,
)
from .capabilities import RenderCapabilities
from .errors import RenderInputError, ValidationIssue
from .feebee import (
    EFeebeeMaterialType,
    EFeebeeShading,
    EFeebeeTextureSampler,
    Feebee,
    FeebeeColourShader,
    FeebeeConfig,
    FeebeeMaterial,
    FeebeeTextureShader,
)
from .imagedef2d import ImageDef2D, ImageDefOpts
from .imagetools import (
    EImageType,
    calculate_edge_function,
    calculate_elem_bound_box_high,
    calculate_elem_bound_box_low,
    format_image_number,
    save_image,
)
from .imagewarp2d import IImageWarp2D
from .light import ELightType, Light
from .mesh import EElementType, Mesh2D, Mesh3D
from .meshops import mesh2d_from_simdata, mesh3d_from_simdata
from .pxint2d import (
    AdditiveSpeckles,
    AnalyticRule,
    Eggbox,
    EPxIntMapping,
    GaussianPSF,
    GaussRule,
    PixIntGrid2D,
    PixIntSpeck2D,
    PxInt2DOpts,
    RectRule,
    quantise_image,
)
from .renderer3d import IRenderer3D
from .result import ImageWarpResult, RenderResult
from .riley import Riley, to_native_camera, to_native_mesh
from .rileyshader import (
    RileyFunctionShader,
    RileyNodalShader,
    RileyTextureShader,
)
from .scene import Scene2D, Scene3D
from .uvtools import (
    EUVFit,
    EUVOrigin,
    EUVPlane,
    UVPlane,
    UVTransform,
    pixels_to_uvs,
    project_uvs_planar,
    project_uvs_planar_centered,
    project_uvs_planar_pixels,
    transform_uvs,
    uvs_to_pixels,
)

__all__ = [
    "AdditiveSpeckles",
    "AnalyticRule",
    "Blender",
    "BlenderCalibrationData",
    "BlenderCalibrationTarget",
    "BlenderConfig",
    "BlenderImageShader",
    "BlenderMaterial",
    "BlenderTextureShader",
    "Camera",
    "Camera2D",
    "CameraStereo",
    "EBlenderDevice",
    "EBlenderEngine",
    "EDistortionModel",
    "EElementType",
    "EFeebeeMaterialType",
    "EFeebeeShading",
    "EFeebeeTextureSampler",
    "EImageType",
    "ELightType",
    "EUVFit",
    "EUVOrigin",
    "EUVPlane",
    "EPSFType",
    "EPxIntMapping",
    "Eggbox",
    "Feebee",
    "FeebeeColourShader",
    "FeebeeConfig",
    "FeebeeMaterial",
    "FeebeeTextureShader",
    "GaussRule",
    "GaussianPSF",
    "IImageWarp2D",
    "IRenderer3D",
    "ImageDef2D",
    "ImageDefOpts",
    "ImageWarpResult",
    "Light",
    "Mesh2D",
    "Mesh3D",
    "PixIntGrid2D",
    "PixIntSpeck2D",
    "PxInt2DOpts",
    "RectRule",
    "RenderCapabilities",
    "RenderInputError",
    "RenderResult",
    "Riley",
    "to_native_camera",
    "to_native_mesh",
    "RileyFunctionShader",
    "RileyNodalShader",
    "RileyTextureShader",
    "Scene2D",
    "Scene3D",
    "ValidationIssue",
    "UVPlane",
    "UVTransform",
    "average_subpixel_image",
    "blender_available",
    "blender_camera_from_resolution",
    "blender_field_of_view",
    "blender_gpu_available",
    "blender_mm_per_pixel",
    "calculate_edge_function",
    "calculate_elem_bound_box_high",
    "calculate_elem_bound_box_low",
    "calibration_image_count",
    "crop_image_rectangle",
    "faceon_stereo_cameras",
    "focal_length_from_resolution",
    "format_image_number",
    "mesh2d_from_simdata",
    "mesh3d_from_simdata",
    "pixel_grid_leng",
    "pixel_vec_leng",
    "pixels_to_uvs",
    "project_uvs_planar",
    "project_uvs_planar_centered",
    "project_uvs_planar_pixels",
    "quantise_image",
    "raise_if_blender_unavailable",
    "render_calibration_images",
    "save_image",
    "subpixel_grid_leng",
    "subpixel_vec_leng",
    "symmetric_stereo_cameras",
    "transform_uvs",
    "uvs_to_pixels",
]
