#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from pyvale.sensorsim.imagetools import EImageType, ImageTools


@dataclass(slots=True)
class RasterOpts:

    image_min_frac: float = 0.0
    image_max_frac: float = 1.0
    image_background_frac: float = 0.5
    image_bits: int = 16
    image_type: EImageType = EImageType.TIFF

    subsample: int = 2
    parallel: int = 4

    force_static: bool = False
    save_depth_array: bool = False
    save_depth_image: EImageType | None = None
    save_image_array: bool = False

    def __post_init__(self) -> None:
        assert self.image_bits > 0, "Image bits must be greater than 0."
        assert self.subsample > 0, "Subsampling must be larger than 0."
        assert self.parallel > 0, "Number of parallel renders must be larger than 0"

def save_raster(save_file: Path,
                image_buff: np.ndarray,
                depth_buff: np.ndarray,
                opts: RasterOpts) -> None:

    ImageTools.scale_digitise_save(save_file,
                                   image_buff,
                                   opts.image_type,
                                   opts.image_bits,
                                   opts.image_min_frac,
                                   opts.image_max_frac,
                                   opts.image_background_frac)

    if opts.save_image_array:
        np.save(save_file.with_suffix(".npy"),image_buff)

    if opts.save_depth_image is not None:
        depth_path = save_file.with_stem(f"{save_file.stem}_depth")
        ImageTools.scale_digitise_save(depth_path,
                                       depth_buff,
                                       opts.save_depth_image)

    if opts.save_depth_array:
        depth_path = save_file.with_stem(f"{save_file.stem}_depth")
        np.save(depth_path.with_suffix(".npy"),depth_buff)