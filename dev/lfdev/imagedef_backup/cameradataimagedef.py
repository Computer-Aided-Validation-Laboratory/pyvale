"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass, field
import warnings
import numpy as np

@dataclass(slots=True)
class CameraDataImageDef2D:
    pixels_count: np.ndarray | None = None
    m_per_px: float = 1.0e-3
    bits: int = 8
    roi_cent_world: np.ndarray | None = None

    field_of_view: np.ndarray = field(init=False)
    dynamic_range: int = field(init=False)

    def __post_init__(self) -> None:

        if self.pixels_count is None:
            self.pixels_count = np.array((1000,1000),dtype=np.int32)

        if self.roi_cent_world is None:
            self.roi_cent_world = np.array((0.0,0.0,0.0),dtype=np.float64)

        self.field_of_view = self.m_per_px*(self.pixels_count.astype(np.float64))
        self.dynamic_range = 2**self.bits



class CameraImageDef:

    def __init__(self,
                 num_px: np.ndarray | None = None,
                 bits: int = 8,
                 m_per_px: float = 1.0e-3) -> None:

        if num_px is None:
            self._num_px = np.array((1000,1000))
        else:
            self._num_px = num_px

        self._bits = bits
        self._m_per_px = m_per_px

        self._fov = self._m_per_px*self._num_px
        self._dyn_range = 2**self._bits
        self._background = int(self._dyn_range/2)

        self._roi_cent = (True,True)
        self._roi_len = self._fov
        self._roi_loc = np.array((0.0,0.0))
        self._coord_offset = np.array((0.0,0.0))

    @property
    def num_px(self) -> np.ndarray:
        return self._num_px

    @property
    def m_per_px(self) -> float:
        return self._m_per_px

    @property
    def fov(self) -> np.ndarray:
        return self._fov

    @property
    def bits(self) -> int:
        return self._bits

    @property
    def dyn_range(self) -> int:
        return self._dyn_range

    @property
    def background(self) -> int:
        return self._background

    @property
    def roi_len(self) -> np.ndarray:
        return self._roi_len

    @property
    def roi_loc(self) -> np.ndarray:
        return self._roi_loc

    @property
    def roi_cent(self) -> tuple[bool,bool]:
        return self._roi_cent

    @property
    def coord_offset(self) -> np.ndarray:
        return self._coord_offset

    @num_px.setter
    def num_px(self, in_px: np.ndarray) -> None:

        if (in_px.ndim != 1) and (in_px.size != 2):
            raise ValueError('Specified camera pixels must be a 1x2 numpy array')

        if (in_px[0] < 1) or (in_px[1] < 1):
            raise ValueError('Camera must have 1 or more pixels in each dimension')

        self._num_px = in_px
        self._fov = self._m_per_px*self._num_px

    @bits.setter
    def bits(self,in_bits: int) -> None:

        if in_bits < 1:
            raise ValueError('Specified number of camera bits must be greater than 0')

        self._bits = in_bits
        self._dyn_range = 2**self._bits
        self._background = self._dyn_range*0.5

    @background.setter
    def background(self, background: int) -> None:
        self._background = background

    @m_per_px.setter
    def m_per_px(self,in_calib):

        if in_calib < 0.0:
            raise ValueError('m to pixel conversion must be greater than 0'
                             )
        self._m_per_px = in_calib
        self._fov = self._m_per_px*self._num_px

    @roi_len.setter
    def roi_len(self,in_len: np.ndarray) -> None:

        self._roi_len = in_len
        self._cent_roi()

    @roi_loc.setter
    def roi_loc(self,in_loc: np.ndarray) -> None:

        if np.sum(self._roi_cent) > 0:
            warnings.warn('ROI is centered, cannot set ROI location. Update centering flags to adjust.')
        else:
            self._roi_loc = in_loc

    @roi_cent.setter
    def roi_cent(self,in_flags: tuple[bool,bool]) -> None:
        self._roi_cent = in_flags
        self._cent_roi()

    @coord_offset.setter
    def coord_offset(self,in_offset: np.ndarray) -> None:
        self._coord_offset = in_offset
        self._cent_roi()

    def _cent_roi(self) -> None:
        if self._roi_cent[0] is True:
            self._roi_loc[0] = (self._fov[0] - self._roi_len[0])/2 + self._coord_offset[0]
        if self._roi_cent[1] is True:
            self._roi_loc[1] = (self._fov[1] - self._roi_len[1])/2 + self._coord_offset[1]
