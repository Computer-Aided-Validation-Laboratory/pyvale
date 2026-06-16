# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from dataclasses import dataclass
from enum import Enum, IntEnum

# ================================================================================
# ENUMS WITH GENERAL RENDERING OPTIONS
# ================================================================================

# Enum to specify render type to be able to let user pick between static and dynamic images
# Would make more sense to be in rtmain, but then we suffer from circular imports
class RenderType(Enum):
    STATIC = 0
    DYNAMIC = 1

# Enum to specify the texture sampler type
# Must match the enum in rtcolorsampling.h on the C++ side
class TextureSampler(IntEnum):
    NEAREST_NEIGHBOUR = 0
    LANCZOS_2 = 1
    LANCZOS_3 = 2
    CATMULL_ROM = 3
    MITCHELL_NETRAVALI = 4
    BSPLINE = 5
    QUINTIC_SPLINE = 6

# Enum to specify which normals are used for shading
class ShadingType(IntEnum):
    FLAT = 0 # Shade with geometric normals for all elements
    BLENDED = 1 # Use angle-avg node normals for TRI3 and QUAD4, Jacobians for curved elements
    ANGLE_AVG_BLENDED = 2 # Angle-avg node normals for all elements

# ================================================================================
# OUTPUT FORMAT
# ================================================================================

# Ray tracer output type
# Enums so users cannot pass weird values
class OutputFormat(IntEnum):
    IMG_PPM = 0
    IMG_TIFF_8BIT = 1
    IMG_TIFF_16BIT = 2
    IMG_BMP_8BIT = 3
    IMG_BMP_24BIT = 4
    #NP_BUFFER = 5 # Not implemented yet; we could stash it in ImageFormat and just allow all options (or something like this)

class BitDepth(IntEnum):
    BIT_8 = 8
    BIT_10 = 10
    BIT_12 = 12
    BIT_16 = 16

class ChannelCount(IntEnum):
    MONO = 1
    RGB = 3
    #RGBA = 4 # Maybe someday

FORMAT_ALLOWED_BIT_DEPTHS: dict[OutputFormat, set[BitDepth]] = {
    OutputFormat.IMG_PPM: {
        BitDepth.BIT_8},
    OutputFormat.IMG_TIFF_8BIT: {
        BitDepth.BIT_8},
    # TIFF-16 allows for many bit-depths
    OutputFormat.IMG_TIFF_16BIT: {
        BitDepth.BIT_8,
        BitDepth.BIT_10,
        BitDepth.BIT_12,
        BitDepth.BIT_16},
    OutputFormat.IMG_BMP_8BIT: {
        BitDepth.BIT_8},
    OutputFormat.IMG_BMP_24BIT: {
        BitDepth.BIT_8}
}

FORMAT_ALLOWED_CHANNEL_COUNTS: dict[OutputFormat, set[ChannelCount]] = {
    OutputFormat.IMG_PPM: {ChannelCount.RGB,},
    OutputFormat.IMG_TIFF_8BIT: {
        ChannelCount.MONO,
        ChannelCount.RGB},
    OutputFormat.IMG_TIFF_16BIT: {
        ChannelCount.MONO,
        ChannelCount.RGB},
    OutputFormat.IMG_BMP_8BIT: {
        ChannelCount.MONO},
    OutputFormat.IMG_BMP_24BIT: {
        ChannelCount.RGB},
}

FORMAT_FILE_BITS_PER_CHANNEL: dict[OutputFormat, int] = {
    OutputFormat.IMG_PPM: 8,
    OutputFormat.IMG_TIFF_8BIT: 8,
    OutputFormat.IMG_TIFF_16BIT: 16,
    OutputFormat.IMG_BMP_8BIT: 8,
    OutputFormat.IMG_BMP_24BIT: 8}
 
 # Frozen = immutable => No need to fiddle with setattr later
@dataclass(slots=True, frozen=True)
class ImageFormat:
    """
    Immutable description of how image data should be exported.

    Parameters
    ----------
    output_format: OutputFormat
        File/container format used for export.
    bit_depth: BitDepth
        Effective source bit depth per channel in the image.
        Example: 12-bit data written into TIFF-16 should use BitDepth.BIT_12.
    channel_count: ChannelCount
        Number of image channels.
    grayscale: bool
        Whether to render grayscale image or not. This is different to setting channel count.

    Raises:
    -------
    ValueError:
        If the number of channels or bit depth exceed the maximum allowance for the chosen file type.

    Notes
    -----
    - TIFF 16-bit accepts effective source depths of 8, 10, 12, or 16 bits, but the written TIFF container still uses 16 bits per sample.
    - BMP 8-bit here means indexed grayscale (MONO only).
    - BMP 24-bit here means RGB with 8 bits per channel.
    """

    output_format: OutputFormat = OutputFormat.IMG_BMP_8BIT
    bit_depth: BitDepth = BitDepth.BIT_8
    channel_count: ChannelCount = ChannelCount.MONO
    grayscale: bool = False

    def __post_init__(self) -> None:
        # Check allowed bit depth vs. what is set
        allowed_depths = FORMAT_ALLOWED_BIT_DEPTHS[self.output_format]
        if self.bit_depth not in allowed_depths:
            allowed_depth_vals = ", ".join(str(depth.value) for depth in sorted(allowed_depths, key=int))
            raise ValueError(
                f"{self.output_format.name} supports source bit depths "
                f"{{{allowed_depth_vals}}} bits per channel, "
                f"but got {self.bit_depth.value}.")

        # Check allowed channel count vs. what is set
        allowed_channels = FORMAT_ALLOWED_CHANNEL_COUNTS[self.output_format]
        if self.channel_count not in allowed_channels:
            allowed_channel_vals = ", ".join(str(channel.value) for channel in sorted(allowed_channels, key=int))
            raise ValueError(
                f"{self.output_format.name} supports channel counts "
                f"{{{allowed_channel_vals}}}, but got {self.channel_count.value}.")

       # Grayscale constraint for BMP 8-bit; for everything else, we allow grayscale RGB
        if self.output_format == OutputFormat.IMG_BMP_8BIT and not self.grayscale:
            raise ValueError(
                "IMG_BMP_8BIT is implemented as an 8-bit grayscale palette BMP and requires grayscale=True.")

    # Mini functions that display the information about the format so it is clear as day
    def file_bits_per_channel(self) -> int:
        return FORMAT_FILE_BITS_PER_CHANNEL[self.output_format]

    def is_grayscale(self) -> bool:
        return self.channel_count == ChannelCount.MONO

    def is_rgb(self) -> bool:
        return self.channel_count == ChannelCount.RGB

    def bytes_per_sample_on_disk(self) -> int:
        return self.file_bits_per_channel() // 8

    def bytes_per_pixel_on_disk(self) -> int:
        return self.bytes_per_sample_on_disk() * int(self.channel_count)

    def supports_source_bit_depth(self, depth: BitDepth) -> bool:
        return depth in FORMAT_ALLOWED_BIT_DEPTHS[self.output_format]

    def supports_channel_count(self, channels: ChannelCount) -> bool:
        return channels in FORMAT_ALLOWED_CHANNEL_COUNTS[self.output_format]

    def describe(self) -> str:
        return (
            f"ImageFormat(output_format={self.output_format.name}, "
            f"bit_depth={self.bit_depth.value}, "
            f"channel_count={self.channel_count.value}, "
            f"file_bits_per_channel={self.file_bits_per_channel()}, "
            f"extension='{self.file_extension()}')")