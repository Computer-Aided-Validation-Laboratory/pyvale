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
    def load_data(self) -> ExpData:
        pass


def load_exp_data(loaders: dict[str,IExpLoader]) -> dict[str,ExpData]:
    exp_data = {}    
    for kk,ll in loaders.items():
        exp_data[kk] = ll.load_data() 
         
    return exp_data
