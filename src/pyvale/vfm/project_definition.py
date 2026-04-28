from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import enum
import numpy as np
import numpy.typing as npt

from pyvale.vfm.mechanical_properties import (
    EConstituitiveLaw,
    EParameterName,
    REQUIRED_PARAMETERS,
    KnownParameter,
    MechanicalProperties,
)


LINEAR_HARDENING_PARAMETER_DEFAULTS: dict[EParameterName, tuple[float, float, float]] = {
    EParameterName.ElasticModulus: (190.0e3, 150.0e3, 250.0e3),
    EParameterName.PoissonsRatio: (0.28, 0.2, 0.4),
    EParameterName.YieldStrength: (320.0, 100.0, 1000.0),
    EParameterName.HardeningModulus: (3000.0, 1000.0, 10000.0),
}

DEFAULT_TEST_DATA_PATH = Path(
    "/home/robh/1_Projects/vfmap-numerical-paper/data/"
    "notchedButtWeld_bilin_lin360420S_hom3700H_imDef_1.5/5-testData/test_data.npz"
)

class EEdgeBoundaryCondition(enum.Enum):
    FREE = enum.auto()
    FIXED = enum.auto()
    TRACTION = enum.auto()

@dataclass(slots=True)
class EdgeBoundaryCondition:
    x: EEdgeBoundaryCondition
    y: EEdgeBoundaryCondition

@dataclass(slots=True)
class BoundaryConditions:
    left: EdgeBoundaryCondition
    upper: EdgeBoundaryCondition
    right: EdgeBoundaryCondition
    lower: EdgeBoundaryCondition


@dataclass(slots=True)
class TestData:
    """Common numerical test-data layout used by the toolkit.
    
    Parameters
    ----------
    x : ndarray
        Shape (num_points_y, num_points_x).
        The x coordinates of the measurement points.
    y : ndarray
        Shape (num_points_y, num_points_x).
        The y coordinates of the measurement points.
    specimen_mask : ndarray of bool
        Shape (num_points_y, num_points_x).
        A mask indicating the specimen region (True for points inside the specimen).
    area : ndarray
        Shape (num_points_y, num_points_x).
        The area associated with each measurement point, used for integration.
    strain : ndarray
        Shape (num_timesteps, num_strain_components, num_points_y, num_points_x).
        The measured strain components at each point (e.g., [exx, eyy, exy]).
    force : ndarray
        Shape (num_timesteps,).
        The applied force measurements corresponding to each time step.
    time : ndarray
        Shape (num_timesteps,).
        The time points corresponding to the measurements.
    thickness : float, default=1.0
        The specimen thickness, used for stress calculations.
    source_path : Path or None, optional
        The file path from which the test data was loaded, if applicable.
    boundary_conditions : BoundaryConditions
        The boundary conditions associated with the test data.
    """

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    specimen_mask: npt.NDArray[np.bool_]
    area: npt.NDArray[np.float64]
    strain: npt.NDArray[np.float64]
    force: npt.NDArray[np.float64]
    time: npt.NDArray[np.float64]
    thickness: float
    boundary_conditions: BoundaryConditions
    source_path: Path | None = None

# Exclude TestData from test collection since it's just a data container and not a test case
TestData.__test__ = False


@dataclass(slots=True)
class ParameterDefinition:
    name: EParameterName
    initial_value_type: str = "float"
    initial_value: float | str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    description: str = ""


def load_parameter_initial_value_array(
    parameter_definition: ParameterDefinition,
) -> npt.NDArray[np.float64]:
    """Load a 2D `.npy` field used as a fixed initial value."""

    if parameter_definition.initial_value_type != "2d np array":
        raise ValueError(
            f"Parameter '{parameter_definition.name.name}' does not use a 2d np array."
        )

    initial_value = parameter_definition.initial_value
    if initial_value is None or str(initial_value).strip() == "":
        raise ValueError(
            f"Parameter '{parameter_definition.name.name}' needs a `.npy` path."
        )

    array = np.load(Path(str(initial_value)))
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"Parameter '{parameter_definition.name.name}' expects a 2D `.npy` array, "
            f"got {array.ndim}D."
        )
    return array


def resolve_parameter_initial_value(
    parameter_definition: ParameterDefinition,
) -> float | npt.NDArray[np.float64] | None:
    """Resolve the parameter initial value into a scalar or a loaded array."""

    if parameter_definition.initial_value_type == "2d np array":
        return load_parameter_initial_value_array(parameter_definition)

    initial_value = parameter_definition.initial_value
    if initial_value is None or str(initial_value).strip() == "":
        return None
    return float(initial_value)


def resolve_parameter_initial_value_scalar(
    parameter_definition: ParameterDefinition,
) -> float | None:
    """Resolve a scalar initial value, averaging a 2D map when needed."""

    resolved_value = resolve_parameter_initial_value(parameter_definition)
    if resolved_value is None:
        return None
    if isinstance(resolved_value, np.ndarray):
        return float(np.nanmean(resolved_value))
    return float(resolved_value)


@dataclass(slots=True)
class ParameterisationSpec:
    kind: str
    name: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    source_phase: str | None = None
    source_parameter: str | None = None
    free_dof_groups: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MetricSpec:
    kind: str
    weight: float = 1.0
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimiserSpec:
    kind: str = "least_squares"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PhaseDefinition:
    name: str
    parameterisations: dict[str, list[ParameterisationSpec]] = field(default_factory=dict)
    metrics: list[MetricSpec] = field(default_factory=list)
    optimiser: OptimiserSpec = field(default_factory=OptimiserSpec)
    notes: str = ""


@dataclass(slots=True)
class IdentificationProject:
    name: str = "vfm_project"
    constituitive_law: EConstituitiveLaw = EConstituitiveLaw.LinearHardening
    test_data_path: Path | None = None
    use_gui: bool = False
    parameters: dict[str, ParameterDefinition] = field(default_factory=dict)
    phases: list[PhaseDefinition] = field(default_factory=list)
    notes: str = ""
    project_path: Path | None = None


@dataclass(slots=True)
class PhaseResult:
    phase_name: str
    cost: float
    metric_values: dict[str, float]
    parameter_maps: dict[str, npt.NDArray[np.float64]]
    stress: npt.NDArray[np.float64] | None = None
    equivalent_stress: npt.NDArray[np.float64] | None = None
    yield_map: npt.NDArray[np.bool_] | None = None
    equivalent_plastic_strain: npt.NDArray[np.float64] | None = None
    best_dof_vector: npt.NDArray[np.float64] | None = None
    parameter_states: dict[str, Any] = field(default_factory=dict)

def create_default_project(
    constituitive_law: EConstituitiveLaw = EConstituitiveLaw.LinearHardening,
) -> IdentificationProject:
    parameters: dict[str, ParameterDefinition] = {}

    for parameter_name in REQUIRED_PARAMETERS[constituitive_law]:
        parameters[parameter_name.name] = create_default_parameter_definition(
            constituitive_law,
            parameter_name,
        )

    return IdentificationProject(
        constituitive_law=constituitive_law,
        test_data_path=DEFAULT_TEST_DATA_PATH,
        parameters=parameters,
        phases=[create_default_phase_definition(constituitive_law)],
    )


def create_default_parameter_definition(
    constituitive_law: EConstituitiveLaw,
    parameter_name: EParameterName,
) -> ParameterDefinition:
    defaults = {}
    if constituitive_law is EConstituitiveLaw.LinearHardening:
        defaults = LINEAR_HARDENING_PARAMETER_DEFAULTS

    try:
        initial_value, lower_bound, upper_bound = defaults[parameter_name]
    except KeyError:
        return ParameterDefinition(name=parameter_name)

    return ParameterDefinition(
        name=parameter_name,
        initial_value_type="float",
        initial_value=initial_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def create_default_phase_definition(
    constituitive_law: EConstituitiveLaw,
    phase_index: int = 1,
) -> PhaseDefinition:
    phase_name = f"phase_{phase_index}"

    if constituitive_law is EConstituitiveLaw.LinearHardening:
        return PhaseDefinition(
            name=phase_name,
            parameterisations={
                EParameterName.ElasticModulus.name: [ParameterisationSpec(kind="known")],
                EParameterName.PoissonsRatio.name: [ParameterisationSpec(kind="known")],
                EParameterName.YieldStrength.name: [
                    ParameterisationSpec(
                        kind="homogeneous",
                        options={"initialise_from": "initial_value"},
                    )
                ],
                EParameterName.HardeningModulus.name: [
                    ParameterisationSpec(
                        kind="homogeneous",
                        options={"initialise_from": "initial_value"},
                    )
                ],
            },
            metrics=[
                MetricSpec(
                    kind="sbvf",
                    weight=1.0,
                    options={
                        "virtual_mesh_size": [15, 15],
                        "stress_sensitivity": "total",
                        "traction_edge": 3,
                        "scale_fraction": 0.3,
                    },
                )
            ],
            optimiser=OptimiserSpec(
                kind="least_squares",
                options={"method": "lm", "max_nfev": 200},
            ),
        )

    return PhaseDefinition(
        name=phase_name,
        parameterisations={
            parameter_name.name: [ParameterisationSpec(kind="known")]
            for parameter_name in REQUIRED_PARAMETERS[constituitive_law]
        },
        metrics=[MetricSpec(kind="sbvf", weight=1.0)],
        optimiser=OptimiserSpec(kind="least_squares"),
    )


def build_mechanical_properties_from_project(
    project: IdentificationProject,
) -> MechanicalProperties:
    """Build the resolved material definition used by constitutive updates.

    At this stage each parameter is just a scalar placeholder. During
    identification the active parameter maps are rebuilt from the spatial
    parameterisation state and replace the corresponding placeholders.
    """

    parameters = {}
    for parameter_name in REQUIRED_PARAMETERS[project.constituitive_law]:
        try:
            parameter_definition = project.parameters[parameter_name.name]
        except KeyError as error:
            raise ValueError(
                f"Project is missing parameter '{parameter_name.name}'."
            ) from error

        initial_value = resolve_parameter_initial_value_scalar(parameter_definition)
        if initial_value is None:
            raise ValueError(
                f"Parameter '{parameter_name.name}' needs an initial_value."
            )

        parameters[parameter_name] = KnownParameter(
            value=float(initial_value),
        )

    mechanical_properties = MechanicalProperties(
        constituitive_law=project.constituitive_law,
        parameters=parameters,
    )
    mechanical_properties.validate()
    return mechanical_properties