# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ==============================================================================

from dataclasses import fields
from abc import ABC, abstractmethod

from pyvale.dataio.exceptions import ExpLoadErr
from pyvale.dataio.expdata import ExpData


class IExpLoader(ABC):
    @abstractmethod
    def get_sens_array_key() -> str:
        pass
        
    @abstractmethod
    def load_data(self) -> ExpData:
        pass


def load_exp_data(loaders: list[IExpLoader]) -> ExpData:

    sens_keys = []
    for ll in loaders:
        sens_keys.append(ll.get_sens_array_key())

    if len(sens_keys) != len(set(sens_keys)):
        raise ExpLoadErr(
            "List of experimental data loaders has duplicate sensor keys, check"
            + " sensor keys for all loaders and remove duplicates."
        )

    exp_data = ExpData()
    
    for ll in loaders:
        loaded_data = ll.load_data()

        for ff in fields(exp_data):
            loaded_attr = getattr(loaded_data,ff.name)
            exp_attr = getattr(exp_data,ff.name)
            # Only works because these are both dictionaries
            exp_attr.update(loaded_attr)
           
        # exp_data.fields.update(loaded_data.fields)
        # exp_data.coords.update(loaded_data.coords)
        # exp_data.times.update(loaded_data.times)
        # exp_data.sens_labels.update(loaded_data.sens_labels)

    return exp_data
