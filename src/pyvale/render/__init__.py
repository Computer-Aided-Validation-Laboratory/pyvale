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

from .camera import Camera, Camera2D
from .camera_stereo import CameraStereo
from .camera_tools import CameraTools
from .blender import (
    Blender,
    BlenderCalibrationData,
    BlenderCalibrationTarget,
    BlenderConfig,
    BlenderImageShader,
    BlenderMaterial,
    BlenderTextureShader,
    EBlenderEngine,
    blender_available,
    blender_gpu_available,
    calibration_image_count,
    render_calibration_images,
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
from .imagewarp2d import IImageWarp2D
from .image_tools import EImageType, ImageTools
from .light import ELightType, Light
from .mesh import EElementType, Mesh2D, Mesh3D
from .pxint2d import (
    AdditiveSpeckles,
    AnalyticRule,
    EPxIntMapping,
    Eggbox,
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
from .scene import RenderMesh, RenderScene
from .riley import Riley
from .simdata import mesh2d_from_simdata, mesh3d_from_simdata

__all__ = [
    "Camera",
    "Camera2D",
    "CameraTools",
    "CameraStereo",
    "Blender",
    "BlenderCalibrationData",
    "BlenderCalibrationTarget",
    "blender_available",
    "blender_gpu_available",
    "calibration_image_count",
    "render_calibration_images",
    "BlenderConfig",
    "BlenderImageShader",
    "BlenderMaterial",
    "BlenderTextureShader",
    "EBlenderEngine",
    "AdditiveSpeckles",
    "AnalyticRule",
    "EElementType",
    "EFeebeeMaterialType",
    "EFeebeeShading",
    "EFeebeeTextureSampler",
    "ELightType",
    "EImageType",
    "EPxIntMapping",
    "Eggbox",
    "Feebee",
    "FeebeeColourShader",
    "FeebeeConfig",
    "FeebeeMaterial",
    "FeebeeTextureShader",
    "GaussianPSF",
    "GaussRule",
    "IImageWarp2D",
    "IRenderer3D",
    "ImageDef2D",
    "ImageDefOpts",
    "ImageTools",
    "ImageWarpResult",
    "Light",
    "Mesh2D",
    "Mesh3D",
    "PixIntGrid2D",
    "PixIntSpeck2D",
    "PxInt2DOpts",
    "RectRule",
    "RenderCapabilities",
    "RenderMesh",
    "RenderInputError",
    "RenderResult",
    "RenderScene",
    "Riley",
    "ValidationIssue",
    "mesh2d_from_simdata",
    "mesh3d_from_simdata",
    "quantise_image",
]
