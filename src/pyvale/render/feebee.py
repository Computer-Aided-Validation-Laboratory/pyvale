# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Feebee ray-tracer scene definitions and renderer scaffold.

Feebee is pyvale's future in-process ray-tracing backend.  The public models
in this module are derived from the previous ray-tracer implementation and
form the stable Python boundary for its future compiled backend.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from .capabilities import RenderCapabilities
from .errors import RenderInputError, ValidationIssue
from .mesh import EElementType, Mesh
from .renderer3d import IRenderer3D
from .result import RenderResult
from .scene import RenderScene
from .verifyinput import raise_if_issues, verify_scene_3d


class EFeebeeMaterialType(Enum):
    """Light-transport models supplied by Feebee.

    ``UNLIT`` returns the shader colour without lighting. ``DIFFUSE``,
    ``SPECULAR``, and ``REFRACTIVE`` use Feebee's ray-tracing material paths.
    """

    DIFFUSE = "diffuse"
    SPECULAR = "specular"
    REFRACTIVE = "refractive"
    UNLIT = "unlit"


class EFeebeeShading(Enum):
    """Normal interpolation methods supported by Feebee.

    ``FLAT`` uses geometric normals. ``BLENDED`` uses the element-specific
    normal interpolation used by the original ray tracer. ``ANGLE_AVERAGED``
    uses angle-averaged node normals for every supported surface topology.
    """

    FLAT = "flat"
    BLENDED = "blended"
    ANGLE_AVERAGED = "angle_averaged"


class EFeebeeTextureSampler(Enum):
    """Texture filters planned for Feebee's texture shader."""

    NEAREST_NEIGHBOUR = "nearest_neighbour"
    LANCZOS_2 = "lanczos_2"
    LANCZOS_3 = "lanczos_3"
    CATMULL_ROM = "catmull_rom"
    MITCHELL_NETRAVALI = "mitchell_netravali"
    BSPLINE = "bspline"
    QUINTIC_SPLINE = "quintic_spline"


@dataclass(frozen=True, slots=True)
class FeebeeMaterial:
    """Material properties for one Feebee mesh shader.

    Parameters
    ----------
    material_type : EFeebeeMaterialType, optional
        Ray-transport model for the mesh.
    colour : numpy.ndarray, optional
        RGB multiplier or refractive tint with shape ``(3,)``. Values are
        linear and non-negative; values exceeding one are permitted.
    refractive_index : float, optional
        Material refractive index. It is required for refractive materials.
    priority : int, optional
        Nesting priority used to resolve overlapping refractive volumes.
    is_shell : bool, optional
        Whether the mesh represents a shell rather than a closed solid.
    thickness : float, optional
        Physical shell thickness. It is used only when ``is_shell`` is true.
    """

    material_type: EFeebeeMaterialType = EFeebeeMaterialType.DIFFUSE
    colour: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float64),
    )
    refractive_index: float | None = None
    priority: int = 0
    is_shell: bool = False
    thickness: float = 1.0

    def __post_init__(self) -> None:
        """Store the immutable colour vector as a float array."""
        object.__setattr__(
            self,
            "colour",
            np.ascontiguousarray(self.colour, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class FeebeeColourShader:
    """Face-colour shader with a Feebee material.

    Parameters
    ----------
    colours : numpy.ndarray
        RGB face colours with shape ``(frames, elements, 3)``. A single frame
        can be used for a static mesh.
    material : FeebeeMaterial, optional
        Light-transport properties applied to the face colours.
    """

    colours: np.ndarray
    material: FeebeeMaterial = field(default_factory=FeebeeMaterial)

    def __post_init__(self) -> None:
        """Store contiguous double-precision face-colour data."""
        object.__setattr__(
            self,
            "colours",
            np.ascontiguousarray(self.colours, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class FeebeeTextureShader:
    """Texture shader with nodal UV coordinates and a Feebee material.

    Parameters
    ----------
    uvs : numpy.ndarray
        Texture coordinates with shape ``(node_count, 2)``.
    texture : numpy.ndarray
        Greyscale or RGB texture image with rank two or three.
    material : FeebeeMaterial, optional
        Light-transport properties applied to sampled texture values.
    """

    uvs: np.ndarray
    texture: np.ndarray
    material: FeebeeMaterial = field(default_factory=FeebeeMaterial)

    def __post_init__(self) -> None:
        """Store contiguous texture-coordinate and image arrays."""
        object.__setattr__(self, "uvs", np.ascontiguousarray(self.uvs))
        object.__setattr__(self, "texture", np.ascontiguousarray(self.texture))


@dataclass(frozen=True, slots=True)
class FeebeeConfig:
    """Global options for a Feebee render request.

    Parameters
    ----------
    background_colour : numpy.ndarray, optional
        Linear RGB scene background with shape ``(3,)``.
    scene_refractive_index : float, optional
        Refractive index of the medium that fills the scene.
    antialiasing_samples : int, optional
        Number of samples evaluated for every pixel.
    shading : EFeebeeShading, optional
        Surface-normal interpolation method.
    texture_sampler : EFeebeeTextureSampler, optional
        Filter used by texture shaders.
    max_depth : int or None, optional
        Maximum secondary-ray depth. ``None`` requests Feebee's automatic
        depth selection.
    min_refractive_depth : int or None, optional
        Minimum deterministic depth before refractive paths may terminate.
    output_dir : pathlib.Path or None, optional
        Directory in which Feebee will write rendered image files.
    """

    background_colour: np.ndarray = field(
        default_factory=lambda: np.full(3, 0.7, dtype=np.float64),
    )
    scene_refractive_index: float = 1.0003
    antialiasing_samples: int = 1
    shading: EFeebeeShading = EFeebeeShading.FLAT
    texture_sampler: EFeebeeTextureSampler = EFeebeeTextureSampler.NEAREST_NEIGHBOUR
    max_depth: int | None = None
    min_refractive_depth: int | None = None
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        """Store background colour and output directory in normal form."""
        object.__setattr__(
            self,
            "background_colour",
            np.ascontiguousarray(self.background_colour, dtype=np.float64),
        )
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))


class Feebee(IRenderer3D):
    """Prepare pyvale scenes for the forthcoming Feebee ray tracer.

    The class already validates Feebee-specific shader, material, and scene
    options before any costly scene expansion occurs. The compiled rendering
    backend has not yet been migrated, so :meth:`render` currently raises
    :class:`NotImplementedError` only after successful validation.

    Parameters
    ----------
    config : FeebeeConfig, optional
        Global options for the Feebee render request.
    """

    capabilities = RenderCapabilities(
        element_types=frozenset(EElementType),
        supports_lights=False,
        supports_camera_distortion=False,
        supports_psf=False,
    )

    def __init__(self, config: FeebeeConfig | None = None) -> None:
        """Create a Feebee renderer with global render options."""
        self.config = FeebeeConfig() if config is None else config

    def verify_input(
        self,
        scene: RenderScene,
    ) -> None:
        """Validate a Feebee scene without expanding geometry or rendering.

        Parameters
        ----------
        scene : RenderScene
            Scene containing common meshes with a
            :class:`FeebeeColourShader` or :class:`FeebeeTextureShader`.

        Raises
        ------
        RenderInputError
            If inputs are unsupported or invalid.
        """
        if not isinstance(scene, RenderScene):
            raise TypeError("Feebee requires a RenderScene.")

        meshes = tuple(mesh for mesh in scene.meshes if isinstance(mesh, Mesh))
        issues = list(verify_scene_3d(meshes, scene.cameras, scene.lights))
        if isinstance(self.config, FeebeeConfig):
            issues.extend(_verify_config(self.config))
        else:
            issues.append(ValidationIssue(
                "config", "TYPE", "Expected render.FeebeeConfig.",
            ))

        if scene.lights:
            issues.append(
                ValidationIssue(
                    "lights",
                    "UNSUPPORTED",
                    "Feebee does not support explicit lights yet.",
                ),
            )

        frame_count: int | None = None
        for mesh_index, mesh in enumerate(scene.meshes):
            if not isinstance(mesh, Mesh):
                issues.append(ValidationIssue(
                    f"scene.meshes[{mesh_index}]", "TYPE",
                    "Feebee requires common render.Mesh objects.",
                ))
                continue
            path = f"meshes[{mesh_index}]"
            issues.extend(_verify_mesh(mesh, path))

            mesh_frame_count = _mesh_frame_count(mesh)
            if mesh_frame_count is None or mesh_frame_count == 1:
                continue
            if frame_count is None:
                frame_count = mesh_frame_count
            elif frame_count != mesh_frame_count:
                issues.append(
                    ValidationIssue(
                        path,
                        "FRAME_COUNT",
                        "All animated mesh data must have the same frame count.",
                    ),
                )

        raise_if_issues(tuple(issues))

    def _render(self, scene: RenderScene) -> RenderResult:
        """Reject rendering until the compiled Feebee backend is migrated.

        Parameters
        ----------
        scene : RenderScene
            Previously validated Feebee scene.

        Raises
        ------
        NotImplementedError
            Always, until the Feebee C++ dispatch layer is integrated.
        """
        raise NotImplementedError(
            "Feebee's compiled rendering backend has not been migrated yet.",
        )


def _verify_config(config: FeebeeConfig) -> tuple[ValidationIssue, ...]:
    """Return cheap validation issues for global Feebee options."""
    if not isinstance(config, FeebeeConfig):
        return (
            ValidationIssue("config", "TYPE", "Expected render.FeebeeConfig."),
        )

    issues: list[ValidationIssue] = []
    if (
        config.background_colour.shape != (3,)
        or not np.isfinite(config.background_colour).all()
        or np.any(config.background_colour < 0.0)
    ):
        issues.append(
            ValidationIssue(
                "config.background_colour",
                "VALUE",
                "Expected three finite, non-negative RGB values.",
            ),
        )
    if (
        not _is_positive_finite(config.scene_refractive_index)
    ):
        issues.append(
            ValidationIssue(
                "config.scene_refractive_index",
                "VALUE",
                "Expected a positive finite refractive index.",
            ),
        )
    if config.antialiasing_samples <= 0:
        issues.append(
            ValidationIssue(
                "config.antialiasing_samples",
                "VALUE",
                "Expected a positive sample count.",
            ),
        )
    if not isinstance(config.shading, EFeebeeShading):
        issues.append(
            ValidationIssue("config.shading", "TYPE", "Expected EFeebeeShading."),
        )
    if not isinstance(config.texture_sampler, EFeebeeTextureSampler):
        issues.append(
            ValidationIssue(
                "config.texture_sampler",
                "TYPE",
                "Expected EFeebeeTextureSampler.",
            ),
        )
    issues.extend(_verify_depths(config.max_depth, config.min_refractive_depth))
    return tuple(issues)


def _verify_depths(
    max_depth: int | None,
    min_refractive_depth: int | None,
) -> tuple[ValidationIssue, ...]:
    """Validate ray-depth options used by the original ray tracer."""
    issues: list[ValidationIssue] = []
    for path, value, minimum in (
        ("config.max_depth", max_depth, 1),
        ("config.min_refractive_depth", min_refractive_depth, 0),
    ):
        if value is not None and (not isinstance(value, int) or value < minimum):
            issues.append(
                ValidationIssue(path, "VALUE", f"Expected an integer >= {minimum}."),
            )
    if (
        isinstance(max_depth, int)
        and isinstance(min_refractive_depth, int)
        and max_depth < min_refractive_depth
    ):
        issues.append(
            ValidationIssue(
                "config.max_depth",
                "VALUE",
                "Maximum depth must be at least the refractive depth.",
            ),
        )
    return tuple(issues)


def _verify_mesh(mesh: Mesh, path: str) -> tuple[ValidationIssue, ...]:
    """Return Feebee-specific validation issues for one mesh."""
    issues: list[ValidationIssue] = []
    nodes_per_element = {
        EElementType.TRI3: 3,
        EElementType.TRI6: 6,
        EElementType.QUAD4: 4,
        EElementType.QUAD8: 8,
        EElementType.QUAD9: 9,
    }
    expected_nodes = nodes_per_element.get(mesh.element_type)
    if expected_nodes is None:
        issues.append(
            ValidationIssue(
                path + ".element_type",
                "UNSUPPORTED",
                "Unsupported Feebee element type.",
            ),
        )
    elif mesh.connectivity.ndim == 2 and mesh.connectivity.shape[1] != expected_nodes:
        issues.append(
            ValidationIssue(
                path + ".connectivity",
                "SHAPE",
                "Expected "
                f"{expected_nodes} nodes per {mesh.element_type.value} element.",
            ),
        )

    shader = mesh.shader
    if not isinstance(shader, (FeebeeColourShader, FeebeeTextureShader)):
        issues.append(
            ValidationIssue(
                path + ".shader",
                "OWNERSHIP",
                "Expected a render.feebee shader.",
            ),
        )
        return tuple(issues)

    issues.extend(_verify_material(shader.material, path + ".shader.material"))
    if isinstance(shader, FeebeeColourShader):
        expected_shape = (None, mesh.connectivity.shape[0], 3)
        if (
            shader.colours.ndim != 3
            or shader.colours.shape[1:] != expected_shape[1:]
            or not _is_finite_array(shader.colours)
            or np.any(shader.colours < 0.0)
        ):
            issues.append(
                ValidationIssue(
                    path + ".shader.colours",
                    "SHAPE",
                    "Expected finite non-negative shape (frames, elements, 3).",
                ),
            )
        if (
            mesh.displacements is not None
            and shader.colours.ndim == 3
            and shader.colours.shape[0] not in (1, mesh.displacements.shape[0])
        ):
            issues.append(
                ValidationIssue(
                    path + ".shader.colours",
                    "FRAME_COUNT",
                    "Expected one colour frame or one per displacement frame.",
                ),
            )
    else:
        if shader.uvs.shape != (mesh.coords.shape[0], 2):
            issues.append(
                ValidationIssue(
                    path + ".shader.uvs",
                    "SHAPE",
                    "Expected shape (nodes, 2).",
                ),
            )
        elif not np.isfinite(shader.uvs).all():
            issues.append(
                ValidationIssue(
                    path + ".shader.uvs",
                    "FINITE",
                    "Texture coordinates must be finite.",
                ),
            )
        if shader.texture.ndim not in (2, 3) or not _is_finite_array(shader.texture):
            issues.append(
                ValidationIssue(
                    path + ".shader.texture",
                    "SHAPE",
                    "Expected a finite greyscale or RGB texture image.",
                ),
            )
    return tuple(issues)


def _verify_material(
    material: FeebeeMaterial,
    path: str,
) -> tuple[ValidationIssue, ...]:
    """Return validation issues for Feebee material properties."""
    issues: list[ValidationIssue] = []
    if not isinstance(material.material_type, EFeebeeMaterialType):
        issues.append(
            ValidationIssue(
                path + ".material_type",
                "TYPE",
                "Expected EFeebeeMaterialType.",
            ),
        )
    if (
        material.colour.shape != (3,)
        or not _is_finite_array(material.colour)
        or np.any(material.colour < 0.0)
    ):
        issues.append(
            ValidationIssue(
                path + ".colour",
                "VALUE",
                "Expected three finite, non-negative RGB values.",
            ),
        )
    if material.material_type is EFeebeeMaterialType.REFRACTIVE and (
        material.refractive_index is None
        or not _is_positive_finite(material.refractive_index)
    ):
        issues.append(
            ValidationIssue(
                path + ".refractive_index",
                "VALUE",
                "Refractive materials require a positive finite index.",
            ),
        )
    if not isinstance(material.priority, int):
        issues.append(
            ValidationIssue(path + ".priority", "TYPE", "Expected an integer."),
        )
    if material.is_shell and (
        not _is_positive_finite(material.thickness)
    ):
        issues.append(
            ValidationIssue(
                path + ".thickness",
                "VALUE",
                "Shell materials require a positive finite thickness.",
            ),
        )
    return tuple(issues)


def _is_finite_array(values: np.ndarray) -> bool:
    """Return whether an array can represent only finite numeric values."""
    try:
        return bool(np.isfinite(values).all())
    except TypeError:
        return False


def _is_positive_finite(value: float | int | np.number) -> bool:
    """Return whether a scalar is finite and strictly positive."""
    try:
        return bool(np.isfinite(value) and value > 0.0)
    except TypeError:
        return False


def _mesh_frame_count(mesh: Mesh) -> int | None:
    """Return the animation frame count represented by one valid mesh."""
    if mesh.displacements is not None:
        return mesh.displacements.shape[0]
    if isinstance(mesh.shader, FeebeeColourShader) and mesh.shader.colours.ndim == 3:
        return mesh.shader.colours.shape[0]
    return None


__all__ = [
    "EFeebeeMaterialType",
    "EFeebeeShading",
    "EFeebeeTextureSampler",
    "Feebee",
    "FeebeeColourShader",
    "FeebeeConfig",
    "FeebeeMaterial",
    "FeebeeTextureShader",
]
