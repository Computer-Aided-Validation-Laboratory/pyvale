"""Shader descriptions owned by the Riley render adapter."""

from dataclasses import dataclass, field

import numpy as np
import riley


@dataclass(slots=True, kw_only=True)
class RileyFunctionShader:
    """An analytic Riley shader evaluated from mesh coordinates."""

    builtin: riley.FuncShaderBuiltin = riley.FuncShaderBuiltin.checker
    coord_mode: riley.FuncCoordMode = riley.FuncCoordMode.world_reference
    parameters: riley.FuncShaderParams = field(
        default_factory=riley.FuncShaderParams,
    )
    uvs: np.ndarray | None = None
    bits: int = 8
    scaling: riley.ScaleStrategy = riley.ScaleStrategy.none


@dataclass(slots=True, kw_only=True)
class RileyTextureShader:
    """A Riley image texture and its nodal UV coordinates."""

    uvs: np.ndarray
    texture: np.ndarray
    sample: riley.TextureSample = riley.TextureSample.cubic_catmull_rom
    sample_mode: riley.TextureSampleMode = riley.TextureSampleMode.lut_lerp
    bits: int = 8
    scaling: riley.ScaleStrategy = riley.ScaleStrategy.none


@dataclass(slots=True, kw_only=True)
class RileyNodalShader:
    """A scalar or colour field defined at mesh nodes."""

    field: np.ndarray
    bits: int = 8
    scaling: riley.ScaleStrategy = riley.ScaleStrategy.auto
    scale_over: riley.ScaleOver = riley.ScaleOver.over_frames


__all__ = [
    "RileyFunctionShader",
    "RileyNodalShader",
    "RileyTextureShader",
]
