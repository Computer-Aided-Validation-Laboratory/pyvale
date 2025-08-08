#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
import warnings
from enum import Enum
from pathlib import Path
import numpy as np
import matplotlib.image as mplim
from PIL import Image

class EImageType(Enum):
    TIFF = ".tiff"
    BMP = ".bmp"

    def __str__(self):
        return self.value



class ImageTools:
    @staticmethod
    def load_image_rgb(im_path: Path) -> np.ndarray:
        return mplim.imread(im_path).astype(np.float64)


    @staticmethod
    def load_image_greyscale(im_path: Path) -> np.ndarray:

        input_im = mplim.imread(im_path).astype(np.float64)
        # If we have RGB then get rid of it
        # TODO: make sure this is collapsing RGB to grey scale correctly
        if input_im.ndim > 2:
            input_im = input_im[:,:,0]

        return input_im

    @staticmethod
    def save_image(save_file: Path,
                   image: np.ndarray,
                   image_type: EImageType,
                   bits: int = 16) -> None:

        # TODO: check this flip is needed
        image_save = np.copy(image[::-1,:])
        im = Image.fromarray(_image_to_uint(image_save,bits))
        im.save(save_file.with_suffix(image_type.value))

    @staticmethod
    def scale_digitise_save(save_file: Path,
                            image: np.ndarray,
                            image_type: EImageType,
                            bits: int = 16,
                            min_frac: float = 0.0,
                            max_frac: float = 1.0,
                            background: float = 0.5) -> None:

        image_save = ImageTools.digitise(image,bits,min_frac,max_frac,background)
        # TODO: check this flip is needed
        image_save = image_save[::-1,:]
        im = Image.fromarray(_image_to_uint(image_save,bits))
        im.save(save_file.with_suffix(image_type.value))


    @staticmethod
    def scale(image: np.ndarray, min_frac: float = 0.0, max_frac: float = 1.0) -> np.ndarray:

        im_scale = np.copy(image)
        im_max = np.nanmax(np.nanmax(image,axis=0),axis=0)
        im_min = np.nanmin(np.nanmin(image,axis=0),axis=0)

        # Scale image 0->1
        im_scale = (im_scale - im_min)/(im_max-im_min)
        
        # Scale to between min->max
        im_scale = im_scale*(max_frac-min_frac) + min_frac

        return im_scale

    @staticmethod
    def digitise(image: np.ndarray,
                 bits: int = 16,
                 min_frac: float = 0.0,
                 max_frac: float = 1.0,
                 background_frac: float = 0.5) -> np.ndarray:
        im_dig = ImageTools.scale(image,min_frac,max_frac)
        mask_nan = np.isnan(image)
        im_dig[mask_nan] = background_frac
        im_dig = _image_to_uint(np.round(2**bits*im_dig),bits)
        return im_dig

    @staticmethod
    def add_noise(image: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def get_num_str(im_num: int, width: int , cam_num: int = -1) -> str:
        num_str = str(im_num)
        num_str = num_str.zfill(width)

        if cam_num >= 0:
            num_str = num_str+'_'+str(cam_num)

        return num_str

    @staticmethod
    def get_save_name(cam_num: int,
                      frame_num: int,
                      field_num: int | None,
                      cam_width: int = 2,
                      frame_width: int = 4,
                      field_width: int = 2) -> str:
        cam_str = str(cam_num).zfill(cam_width)
        frame_str = str(frame_num).zfill(frame_width)

        if field_num is None:
            return f"cam{cam_str}_frame{frame_str}"

        field_str = str(field_num).zfill(field_width)
        return f"cam{cam_str}_frame{frame_str}_field{field_str}"


def _image_to_uint(image: np.ndarray, bits: int) -> np.ndarray:
    if (bits > 16) and (bits <= 32):
        return image.astype(np.uint32)

    if (bits > 8) and (bits <= 16):
        return image.astype(np.uint16)

    if (bits > 0) and (bits <= 8):
        return image.astype(np.uint8)

    warnings.warn(f"Number of bits={bits} should be between 0 and 32, defaulting to 16 bits.")
    return image.astype(np.uint16)


