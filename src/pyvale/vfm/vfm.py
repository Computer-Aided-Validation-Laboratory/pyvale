import enum

from pyvale.vfm.constitutive_laws.constitutive_law import ConstitutiveLaw
from pyvale.vfm.identification_phase import IdentificationPhase
from pyvale.vfm.parameter import Parameter


class EIdentificationType(enum.Enum):
    Linear = enum.auto()
    Nonlinear = enum.auto()


# TODO: return type
def vfm(
    constitutive_law: ConstitutiveLaw,
    identification_type: EIdentificationType,
    parameters: dict[str, Parameter],
    identification_phases: dict[int, IdentificationPhase]
):
    print("test")


# TODO: remove when calling this as a module
# if __name__ == "__main__":
    # vfm()
