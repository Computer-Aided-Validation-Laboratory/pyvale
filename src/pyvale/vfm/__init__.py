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
from .metricsbvf import MetricSBVF
from .metricsliceforce import SliceWiseForceReconstructionMetric
from .metricequilibriumgap import EquilibriumGapMetric

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
    CombinedObjectiveBaseline,
    CombinedObjectiveBaselineMode,
    infer_egi_window_length_weights,
)
from .progress import ConsoleProgressReporter, ProgressEvent

from .refinement import (
    BasisAddRemoveRefinement,
    EquilibriumGapBasisGrowthRefinement,
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
    "SliceWiseForceReconstructionMetric",
    "EquilibriumGapMetric",
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
    "CombinedObjectiveBaseline",
    "CombinedObjectiveBaselineMode",
    "infer_egi_window_length_weights",
    "ProgressEvent",
    "ConsoleProgressReporter",
    "IRefinementPolicy",
    "IRefinementAction",
    "SliceMergeSplitRefinement",
    "BasisAddRemoveRefinement",
    "EquilibriumGapBasisGrowthRefinement",
    "VfmRegionOfInterest",
    "convert_mask_to_physical_roi",
]
