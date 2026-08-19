"""Generic workflow orchestration for pyvale studies."""

from .case import (
    EParameterKind,
    ExplicitCases,
    FullFactorial,
    ICaseDesign,
    ParameterValues,
    RandomSampling,
    WorkflowCase,
)
from .config import EFailurePolicy, EWorkflowStorage, WorkflowConfig
from .errors import WorkflowCaseError, WorkflowError
from .executor import (
    IWorkflowExecutor,
    LocalArrayExecutor,
    SlurmConfig,
    SlurmExecutor,
    SlurmSubmission,
)
from .gather import ConvergenceMetric, WorkflowGatherer, plot_signal_to_noise
from .noise import add_grey_level_noise
from .pipeline import (
    FunctionStep,
    IWorkflow,
    IWorkflowStep,
    PipelineWorkflow,
    WorkflowContext,
)
from .result import CaseResult, ECaseStatus, WorkflowDataset
from .runner import WorkflowRunner
from .selector import (
    AreaSelector,
    ESpatialReduction,
    EStrainComponent,
    FullFieldSelector,
    ISpatialSelector,
    LineSelector,
    MaskSelector,
    PointSelector,
    SignalExtraction,
)

__all__ = [
    "AreaSelector", "CaseResult", "ConvergenceMetric", "ECaseStatus",
    "EFailurePolicy",
    "EParameterKind", "ESpatialReduction", "EStrainComponent",
    "EWorkflowStorage", "ExplicitCases", "FullFactorial",
    "FullFieldSelector", "FunctionStep", "ICaseDesign", "ISpatialSelector",
    "IWorkflow", "IWorkflowExecutor",
    "IWorkflowStep", "LineSelector", "MaskSelector", "ParameterValues",
    "PipelineWorkflow",
    "LocalArrayExecutor", "PointSelector", "RandomSampling", "SignalExtraction",
    "SlurmConfig", "SlurmExecutor", "SlurmSubmission", "WorkflowCase",
    "WorkflowCaseError", "WorkflowConfig", "WorkflowError",
    "WorkflowContext",
    "WorkflowDataset", "WorkflowGatherer", "WorkflowRunner",
    "add_grey_level_noise", "plot_signal_to_noise",
]
