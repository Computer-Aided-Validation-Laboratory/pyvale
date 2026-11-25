from dataclasses import dataclass
from copy import copy
from typing import Self

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import (
    EIdentificationType,
    IConstitutiveLaw,
)
from pyvale.vfm.hardening import IHardeningFunction
from pyvale.vfm.radialreturn import (
    RadialReturnPreparedInputs,
    prepare_radial_return_inputs,
    radial_return,
)


@dataclass(slots=True)
class IsotropicVonMisesElastoplasticity(IConstitutiveLaw):
    """
    Isotropic von Mises (J2) elasto-plasticity in plane stress.

    Combines linear isotropic elasticity with a J2 yield surface and the
    supplied isotropic ``hardening_function``. The required parameters are
    ``elastic_modulus`` and ``poissons_ratio`` plus whichever parameters the
    hardening law needs. The label arguments allow the elastic parameters to
    be renamed if your parameter dictionary uses different keys

    Attributes
    ----------
    hardening_function : IHardeningFunction
        The hardening function to use for the plasticity model.
    elastic_modulus_label : str
        The label for the elastic modulus parameter in the constitutive parameter maps.
    poissons_ratio_label : str
        The label for the Poisson's ratio parameter in the constitutive parameter maps.
    error_tolerance : float
        The error tolerance for the newton-raphson optimisation of the radial return algorithm.
    """

    hardening_function: IHardeningFunction
    elastic_modulus_label: str
    poissons_ratio_label: str
    error_tolerance: float
    cache_radial_return: bool
    _prepared_inputs: RadialReturnPreparedInputs | None

    def __init__(
        self,
        hardening_function: IHardeningFunction,
        elastic_modulus_label: str | None = None,
        poissons_ratio_label: str | None = None,
        *,
        error_tolerance: float = 1.0e-8,
        cache_radial_return: bool = True,
    ) -> None:
        self.hardening_function = hardening_function

        if elastic_modulus_label is not None:
            self.elastic_modulus_label = elastic_modulus_label
        else:
            self.elastic_modulus_label = "elastic_modulus"

        if poissons_ratio_label is not None:
            self.poissons_ratio_label = poissons_ratio_label
        else:
            self.poissons_ratio_label = "poissons_ratio"

        if not np.isfinite(error_tolerance) or error_tolerance <= 0.0:
            raise ValueError("error_tolerance must be finite and greater than zero")
        self.error_tolerance = float(error_tolerance)
        self.cache_radial_return = bool(cache_radial_return)
        self._prepared_inputs = None

    def prepare_for_optimisation(
        self,
        strain: npt.NDArray[np.float64],
        *,
        error_tolerance: float,
        fixed_elastic_parameter_maps: dict[str, npt.NDArray[np.float64]] | None = None,
        cache_radial_return: bool = True,
    ) -> Self:
        """Return a phase-local law with reusable radial-return inputs.

        The returned shallow copy is used only for optimiser candidate
        evaluations. Constitutive configuration, including the read-only
        hardening evaluator, is shared; the optimisation tolerance and
        prepared radial-return inputs belong to the returned law. Set
        ``cache_radial_return`` to ``False`` to retain the optimisation
        tolerance without preparing or using cached inputs.

        The original law therefore retains its reconstruction tolerance for
        final stress maps and ordinary direct calls. This allows the optimiser
        to use a more relaxed tolerance for speed, while the final stress maps are
        calculated with a more stringent tolerance for accuracy.

        The prepared inputs are used to accelerate the radial return algorithm
        by precomputing invariant quantities that depend on the strain field
        and elastic parameters. If the elastic parameters are fixed, they can
        be provided in `fixed_elastic_parameter_maps` to avoid redundant calculations.

        A slice-wise phase calls this method first for the full field and then
        once per local slice. A shallow copy avoids duplicating the existing
        full-field cache; the local cache prepared below simply replaces it on
        the new law.
        """

        prepared_law = copy(self)
        if not np.isfinite(error_tolerance) or error_tolerance <= 0.0:
            raise ValueError("error_tolerance must be finite and greater than zero")
        prepared_law.error_tolerance = float(error_tolerance)
        prepared_law.cache_radial_return = bool(cache_radial_return)
        elastic_modulus = None
        poissons_ratio = None

        # If phase-local fixed elastic parameters have been provided,
        # unpack elastic modulus and Poissons ratio.
        if fixed_elastic_parameter_maps is not None:
            elastic_modulus = fixed_elastic_parameter_maps.get(
                self.elastic_modulus_label
            )
            poissons_ratio = fixed_elastic_parameter_maps.get(
                self.poissons_ratio_label
            )
        if prepared_law.cache_radial_return:
            # Prepare invariant inputs for the radial return algorithm, which
            # depend on the strain field and, optionally, the elastic parameters.
            prepared_law._prepared_inputs = prepare_radial_return_inputs(
                strain,
                elastic_modulus=elastic_modulus,
                poissons_ratio=poissons_ratio,
            )
        else:
            prepared_law._prepared_inputs = None
        return prepared_law

    def get_identification_type(self) -> EIdentificationType:
        return EIdentificationType.Nonlinear

    def get_required_parameters(self) -> list[str]:
        params = [self.elastic_modulus_label, self.poissons_ratio_label]
        params.extend(self.hardening_function.get_required_parameters())
        return params

    def calculate_stress(
        self,
        strain: npt.NDArray[np.float64],
        constitutive_parameter_maps: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        prepared_inputs = self._prepared_inputs if self.cache_radial_return else None
        if (
            prepared_inputs is not None
            and prepared_inputs.strain_shape != tuple(strain.shape)
        ):
            # Slice-wise solvers may call this phase-local law with a smaller
            # local field. Its full-field prepared inputs are not applicable;
            # radial_return will create inputs for this direct call instead.
            prepared_inputs = None

        stress, _, _, _ = radial_return(
            strain,
            constitutive_parameter_maps,
            constitutive_parameter_maps[self.elastic_modulus_label],
            constitutive_parameter_maps[self.poissons_ratio_label],
            self.hardening_function,
            error_tolerance=self.error_tolerance,
            prepared_inputs=prepared_inputs,
        )

        return stress
