from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml


class InputDataFile(ABC):
    @abstractmethod
    def load(self) -> Any:
        pass


@dataclass(slots=True)
class TxtFile(InputDataFile):
    path: str

    def load(self) -> np.ndarray:
        return np.genfromtxt(self.path)


@dataclass(slots=True)
class SingleFieldCsvFile(InputDataFile):
    path: str

    def load(self) -> np.ndarray:
        return np.genfromtxt(
            self.path,
            delimiter=","
        )


@dataclass(slots=True)
class MultiFieldCsvFile(InputDataFile):
    path: str
    header_name: str

    def load(self) -> np.ndarray:
        content = np.genfromtxt(
            self.path,
            delimiter=",",
            names=True
        )

        return content[self.header_name]


CsvFile = (SingleFieldCsvFile | MultiFieldCsvFile)


@dataclass(slots=True)
class NpyFile(InputDataFile):
    path: str

    def load(self) -> np.ndarray:
        content = np.asarray(
            np.load(self.path),
            dtype=np.float64
        )

        return content


# @dataclass(slots=True)
# class NpzFile(InputDataFile):
#     path: str

#     def load(self) -> np.ndarray:
#         return np.asarray(
#             np.load(self.path),
#             dtype=np.float64
#         )


@dataclass(slots=True)
class YamlFile(InputDataFile):
    path: str

    def load(self):
        return yaml.safe_load(self.path)


# @dataclass(slots=True)
# class H5File:
#     path: Path
