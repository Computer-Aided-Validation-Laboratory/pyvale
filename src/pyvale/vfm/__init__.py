#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

from .identification import run_identification
from .identificationconfig import IdentificationConfig, IdentificationPhase
from .identificationresult import (
    IdentificationHistory,
    IdentificationResult,
    ParameterisationSnapshot,
    PhaseResult,
    PhaseSnapshot,
    RefinementEvent,
    SolveResult,
    load_identification_result,
)

from .inputdata import process_input_data
from .inputdataassembled import AssembledDataConfig, load_assembled_data
from .inputdataconfig import AnsysConfig, InputDataConfig, MooseConfig
from .experimentdata import (
    BoundaryConditions,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)

from .constlaw import IConstitutiveLaw
from .constlaws import IsotropicVonMisesElastoplasticity
from .constparam import ConstitutiveParameter
from .dof import DegreeOfFreedom
from .hardening import (
    HardeningLinear,
    HardeningLudwik,
    HardeningSwift,
    HardeningVoce,
    IHardeningFunction,
)

from .spatialparam import ISpatialParameterisation
from .spatialparamknown import SpatialParameterisationKnown
from .spatialparamhomogeneous import SpatialParameterisationHomogeneous
from .spatialparambasisfuncs import (
    BasisFunctionKernelBivariateSPD,
    SpatialParameterisationBasisFunction,
    SupportBasis,
)
from .spatialparamslicewise import (
    SliceWiseSpatialParameterisation,
    SupportSlice,
)
from .slicewise_utils import (
    SliceAreaPartition,
    SliceConfig,
    resolve_cell_aligned_slice_boundaries,
)

from .optimiser import IOptimiser
from .optimiserleastsquares import OptimiserLeastSquares
from .optimiserpatternsearch import OptimiserPatternSearch
from .optimiserslicewiseindependent import SliceWiseIndependentLeastSquares

from .metric import IMetric, MetricResult
from .metricsbvf import (
    MetricSBVF,
    calculate_local_parameter_stress_sensitivity,
    calculate_parameter_stress_sensitivities,
)
from .metricsliceforce import SliceWiseForceReconstructionMetric
from .metricequilibriumgap import EquilibriumGapMetric
from .spatialweighting import (
    SensitivitySpatialWeightingConfig,
    SensitivitySpatialWeights,
    calculate_sensitivity_spatial_weights,
    resolve_sensitivity_spatial_weights,
)

from .objectivefunc import (
    IObjectiveFunction,
    IScalarObjectiveFunction,
    IVectorObjectiveFunction,
)
from .objectivefuncscalar import ScalarFirstResultPassthrough, ScalarFirstResultRms
from .objectivefuncvector import (
    VectorConcatenateObjective,
    VectorFirstResultPassthrough,
    VectorWeightedObjective,
)
from .equilibriumgapaggregation import (
    EquilibriumGapAggregationResult,
    aggregate_equilibrium_gap_results,
    calculate_combined_equilibrium_gap_spatial_rms,
    combine_equilibrium_gap_maps,
    infer_window_area_weights,
)
from .objectivefuncfreandegi import (
    ForceAndEquilibriumGapObjectiveResult,
    ScalarForceAndEquilibriumGapObjective,
)
from .objectivefunccombinedfreegi import (
    CombinedForceAndEquilibriumGapObjective,
    CombinedForceAndEquilibriumGapObjectiveResult,
    CombinedObjectiveResidualCotangents,
    CombinedObjectiveBaseline,
    CombinedObjectiveBaselineMode,
    infer_egi_window_length_weights,
)
from .progress import ConsoleProgressReporter, ProgressEvent
from .egisupports import (
    EgiSupportBankConfig,
    EgiSupportEvidence,
    EgiSupportInformationEvidence,
    EgiSupportInformationSelection,
    EgiSupportInformationSelectionConfig,
    EgiSupportInformationSweep,
    EgiSupportSelection,
    EgiSupportSelectionConfig,
    EgiSupportSweepResult,
    EgiSignalEvidence,
    EgiSignalSelection,
    EgiSignalSelectionConfig,
    EgiSignalSweep,
    PhysicalEgiSupport,
    analyse_egi_support_sweep,
    analyse_egi_signal_sweep,
    analyse_egi_support_information,
    generate_physical_egi_support_bank,
    generate_odd_pixel_egi_support_bank,
    resolve_physical_egi_supports,
    select_sparse_egi_supports,
    select_log_spaced_egi_supports,
    select_information_egi_supports,
)
from .phasepreparation import (
    AutomaticEgiSupportPreparation,
    FixedEgiSupportPreparation,
    SimpleEgiSupportPreparation,
    UserFineEgiSupportPreparation,
    IPhasePreparation,
    PhasePreparationContext,
    PhasePreparationResult,
)
from .objectivefuncsensitivityinformation import (
    SensitivityInformationObjective,
    SensitivityInformationObjectiveConfig,
    SensitivityInformationObjectiveResult,
)
from .objectivefuncsensitivitygated import (
    SensitivityGatedEgiObjective,
    SensitivityGatedObjectiveConfig,
    SensitivityGatedObjectiveResult,
)
from .solvepreparation import (
    SolveDegreeOfFreedom,
    SolvePreparationContext,
)
from .residualblocks import (
    CanonicalResidualLayout,
    CanonicalResidualVector,
    PreparedResidualBlock,
    ResidualBlockSpec,
    prepare_canonical_residual_layout,
)
from .materialprojection import (
    FiniteDifferenceSensitivity,
    NativeDofSensitivityAudit,
    NativeDofSensitivityAuditConfig,
    bound_aware_sensitivity,
    prepare_native_dof_sensitivity_audit,
)

from .refinement import (
    BasisAddRemoveRefinement,
    EquilibriumGapBasisGrowthRefinement,
    SensitivityCorrectionBasisGrowthRefinement,
    IRefinementAction,
    IRefinementPolicy,
    SliceMergeSplitRefinement,
)

from .roi import VfmRegionOfInterest, convert_mask_to_physical_roi

__all__ = [
    "run_identification",
    "IdentificationConfig",
    "IdentificationPhase",
    "IdentificationResult",
    "load_identification_result",
    "IdentificationHistory",
    "PhaseResult",
    "PhaseSnapshot",
    "ParameterisationSnapshot",
    "SolveResult",
    "RefinementEvent",
    "process_input_data",
    "AssembledDataConfig",
    "load_assembled_data",
    "AnsysConfig",
    "MooseConfig",
    "InputDataConfig",
    "ExperimentData",
    "SpecimenGeometry",
    "BoundaryConditions",
    "EdgeConditions",
    "Edge",
    "EEdgeCondition",
    "IConstitutiveLaw",
    "IsotropicVonMisesElastoplasticity",
    "ConstitutiveParameter",
    "DegreeOfFreedom",
    "IHardeningFunction",
    "HardeningLinear",
    "HardeningSwift",
    "HardeningVoce",
    "HardeningLudwik",
    "ISpatialParameterisation",
    "SpatialParameterisationKnown",
    "SpatialParameterisationHomogeneous",
    "SpatialParameterisationBasisFunction",
    "BasisFunctionKernelBivariateSPD",
    "SupportBasis",
    "SliceWiseSpatialParameterisation",
    "SupportSlice",
    "SliceConfig",
    "SliceAreaPartition",
    "resolve_cell_aligned_slice_boundaries",
    "IOptimiser",
    "OptimiserLeastSquares",
    "OptimiserPatternSearch",
    "SliceWiseIndependentLeastSquares",
    "IMetric",
    "MetricResult",
    "MetricSBVF",
    "calculate_parameter_stress_sensitivities",
    "calculate_local_parameter_stress_sensitivity",
    "SliceWiseForceReconstructionMetric",
    "EquilibriumGapMetric",
    "SensitivitySpatialWeightingConfig",
    "SensitivitySpatialWeights",
    "calculate_sensitivity_spatial_weights",
    "resolve_sensitivity_spatial_weights",
    "IObjectiveFunction",
    "IScalarObjectiveFunction",
    "IVectorObjectiveFunction",
    "ScalarFirstResultPassthrough",
    "ScalarFirstResultRms",
    "VectorFirstResultPassthrough",
    "VectorConcatenateObjective",
    "VectorWeightedObjective",
    "EquilibriumGapAggregationResult",
    "aggregate_equilibrium_gap_results",
    "calculate_combined_equilibrium_gap_spatial_rms",
    "combine_equilibrium_gap_maps",
    "infer_window_area_weights",
    "ForceAndEquilibriumGapObjectiveResult",
    "ScalarForceAndEquilibriumGapObjective",
    "CombinedForceAndEquilibriumGapObjective",
    "CombinedForceAndEquilibriumGapObjectiveResult",
    "CombinedObjectiveResidualCotangents",
    "CombinedObjectiveBaseline",
    "CombinedObjectiveBaselineMode",
    "infer_egi_window_length_weights",
    "ProgressEvent",
    "ConsoleProgressReporter",
    "PhysicalEgiSupport",
    "EgiSupportBankConfig",
    "EgiSupportEvidence",
    "EgiSupportInformationEvidence",
    "EgiSupportInformationSweep",
    "EgiSupportInformationSelectionConfig",
    "EgiSupportInformationSelection",
    "EgiSupportSweepResult",
    "EgiSupportSelectionConfig",
    "EgiSupportSelection",
    "EgiSignalEvidence",
    "EgiSignalSweep",
    "EgiSignalSelectionConfig",
    "EgiSignalSelection",
    "resolve_physical_egi_supports",
    "analyse_egi_support_sweep",
    "analyse_egi_signal_sweep",
    "analyse_egi_support_information",
    "generate_physical_egi_support_bank",
    "generate_odd_pixel_egi_support_bank",
    "select_sparse_egi_supports",
    "select_log_spaced_egi_supports",
    "select_information_egi_supports",
    "IPhasePreparation",
    "AutomaticEgiSupportPreparation",
    "FixedEgiSupportPreparation",
    "SimpleEgiSupportPreparation",
    "UserFineEgiSupportPreparation",
    "PhasePreparationContext",
    "PhasePreparationResult",
    "SensitivityInformationObjective",
    "SensitivityInformationObjectiveConfig",
    "SensitivityInformationObjectiveResult",
    "SensitivityGatedEgiObjective",
    "SensitivityGatedObjectiveConfig",
    "SensitivityGatedObjectiveResult",
    "SolveDegreeOfFreedom",
    "SolvePreparationContext",
    "ResidualBlockSpec",
    "PreparedResidualBlock",
    "CanonicalResidualLayout",
    "CanonicalResidualVector",
    "prepare_canonical_residual_layout",
    "FiniteDifferenceSensitivity",
    "NativeDofSensitivityAudit",
    "NativeDofSensitivityAuditConfig",
    "bound_aware_sensitivity",
    "prepare_native_dof_sensitivity_audit",
    "IRefinementPolicy",
    "IRefinementAction",
    "SliceMergeSplitRefinement",
    "BasisAddRemoveRefinement",
    "EquilibriumGapBasisGrowthRefinement",
    "SensitivityCorrectionBasisGrowthRefinement",
    "VfmRegionOfInterest",
    "convert_mask_to_physical_roi",
]
