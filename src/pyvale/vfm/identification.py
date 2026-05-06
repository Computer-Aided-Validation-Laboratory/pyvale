from dataclasses import dataclass
import enum

from pyvale.vfm.constitutive_laws.constitutive_law import ConstitutiveLaw
from pyvale.vfm.identification_phase import IdentificationPhase
from pyvale.vfm.constitutive_parameter import ConstitutiveParameter


class EIdentificationType(enum.Enum):
    Linear = enum.auto()
    Nonlinear = enum.auto()


@dataclass(slots=True)
class Identification():
    constitutive_law: ConstitutiveLaw
    parameters: dict[str, ConstitutiveParameter]
    phases: list[IdentificationPhase]

