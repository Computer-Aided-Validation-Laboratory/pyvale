'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np
import pyvista as pv

import mooseherder as mh

from pyvale.physics.field import (IField,
                                  FieldError,
                                  conv_simdata_to_pyvista,
                                  sample_pyvista)

class VectorField(IField):
    def __init__(self,
                 sim_data: mh.SimData,
                 field_key: str,
                 components: tuple[str,...],
                 spat_dim: int) -> None:

        self._field_key = field_key
        self._components = components
        self._spat_dim = spat_dim

        self._time_steps = sim_data.time
        self._pyvista_grid = conv_simdata_to_pyvista(sim_data,
                                                    components,
                                                    spat_dim)

    def set_sim_data(self, sim_data: mh.SimData) -> None:
        self._time_steps = sim_data.time
        self._time_steps = sim_data.time
        self._pyvista_grid = conv_simdata_to_pyvista(sim_data,
                                                    self._components,
                                                    self._spat_dim)

    def get_time_steps(self) -> np.ndarray:
        return self._time_steps

    def get_visualiser(self) -> pv.UnstructuredGrid:
        return self._pyvista_grid

    def get_all_components(self) -> tuple[str, ...]:
        return self._components

    def get_component_index(self,comp: str) -> int:
        return self._components.index(comp)

    def sample_field(self,
                sample_points: np.ndarray,
                sample_times: np.ndarray | None = None
                ) -> np.ndarray:

        return sample_pyvista(self._components,
                                self._pyvista_grid,
                                self._time_steps,
                                sample_points,
                                sample_times)
