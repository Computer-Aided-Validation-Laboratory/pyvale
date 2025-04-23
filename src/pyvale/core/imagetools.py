"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""
import warnings
from pathlib import Path
import numpy as np
import matplotlib.image as mplim
from PIL import Image


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
    def save_tiff(save_file: Path,
                  image: np.ndarray,
                  bits: int = 16) -> None:
        _image_save(save_file,image,".tiff",bits)

    @staticmethod
    def save_bmp(save_file: Path,
                 image: np.ndarray,
                 bits: int = 16) -> None:
        _image_save(save_file,image,".bmp",bits)

    @staticmethod
    def scale(image: np.ndarray, min: float = 0.0, max: float = 1.0) -> np.ndarray:

        im_scale = np.copy(image)
        im_max = np.max(np.max(image,axis=0),axis=0)
        im_min = np.min(np.min(image,axis=0),axis=0)

        # Scale image 0->1
        im_scale = (im_scale - im_min)/(im_max-im_min)

        # Scale to between min->max
        im_scale = im_scale*(max-min) + min

        return im_scale

    @staticmethod
    def digitise(image: np.ndarray,
                 bits: int,
                 min_frac: float = 0.0,
                 max_frac: float = 1.0) -> np.ndarray:

        im_dig = ImageTools.scale(image,min_frac,max_frac)
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


def _image_to_uint(image: np.ndarray, bits: int) -> np.ndarray:
    if (bits > 16) and (bits <= 32):
        return image.astype(np.uint32)

    if (bits > 8) and (bits <= 16):
        return image.astype(np.uint16)

    if (bits > 0) and (bits <= 8):
        return image.astype(np.uint8)

    warnings.warn(f"Number of bits={bits} should be between 0 and 32, defaulting to 16 bits.")
    return image.astype(np.uint16)

def _image_save(save_file: Path,
                image: np.ndarray,
                ext: str,
                bits: int = 16) -> None:
        # Need to flip image so coords are top left with Y down
        # TODO check this
        image = image[::-1,:]
        im = Image.fromarray(_image_to_uint(image,bits))
        im.save(save_file.with_suffix(ext))
