# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Experimental design through sensor placement optimisation."""

from pyvale.sensopt.designspace import (
    ConstraintSensorCost,
    ConstraintSensorSeparation,
    DesignSpace,
    EDesignAxis,
    IConstraintInput,
    filter_sensor_sites_in_mesh,
)
from pyvale.sensopt.engine import OptResult, SensorOptEngine
from pyvale.sensopt.fieldrecon import FieldReconLinear, FieldReconNearest, IFieldRecon
from pyvale.sensopt.objective import (
    EErrMetric,
    ESignalAxis,
    ESignalReduce,
    ExperimentData,
    IObjFunc,
    IObjNormalisation,
    ObjFieldRecon,
    ObjNormAffine,
    ObjNormLog,
    ObjNormLog1p,
    ObjSensorCost,
    ObjSignal,
    ObjSignalPeakAbs,
    ObjSignalRMS,
    ObjTerm,
    ObjWeighted,
    SignalReduction,
)
from pyvale.sensopt.optimiser import (
    EOptStopCond,
    OptExchange,
    OptExhaustive,
    OptGreedyBackward,
    OptGreedyForward,
    OptStartExhaustive,
    OptStartLayout,
    OptStartRandom,
)

__all__ = [
    "ConstraintSensorCost",
    "ConstraintSensorSeparation",
    "DesignSpace",
    "EDesignAxis",
    "EErrMetric",
    "EOptStopCond",
    "ESignalAxis",
    "ESignalReduce",
    "ExperimentData",
    "FieldReconLinear",
    "FieldReconNearest",
    "IConstraintInput",
    "IFieldRecon",
    "IObjFunc",
    "IObjNormalisation",
    "ObjFieldRecon",
    "ObjNormAffine",
    "ObjNormLog",
    "ObjNormLog1p",
    "ObjSensorCost",
    "ObjSignal",
    "ObjSignalPeakAbs",
    "ObjSignalRMS",
    "ObjTerm",
    "ObjWeighted",
    "OptExchange",
    "OptExhaustive",
    "OptGreedyBackward",
    "OptGreedyForward",
    "OptResult",
    "OptStartExhaustive",
    "OptStartLayout",
    "OptStartRandom",
    "SensorOptEngine",
    "SignalReduction",
    "filter_sensor_sites_in_mesh",
]
