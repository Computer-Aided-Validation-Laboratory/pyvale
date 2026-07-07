"""
Minimal reader for the tightly-packed 12-bit grayscale TIFFs produced by
pyvale's C++ saveTIFF_16bit<MONO> writer (BitDepth::BIT_12).

Why this exists
---------------
The writer stores each pixel as a 12-bit sample packed MSB-first into a
continuous bitstream (2 pixels = 3 bytes, no per-pixel byte alignment).
OpenCV / libtiff read BitsPerSample=12 by pulling 16 bits per sample, so the
decoded values are garbage that drifts across the image -> non-monotonic
convergence, even inside a ROI.

This decoder unpacks the 12-bit samples correctly and returns a uint16 array
of logical codes in [0, 4095]. NumPy 1.x compatible, no tifffile (it depends
on numpy version that is incompatible with other pyvale deps).
"""

import struct
from pathlib import Path

import numpy as np

# TIFF field type -> (struct code, byte size)
_TYPE = {1: ("B", 1), 2: ("c", 1), 3: ("H", 2), 4: ("I", 4), 5: ("I", 4)}


def _read_ifd(raw: bytes):
    if raw[:2] == b"II":
        end = "<"
    elif raw[:2] == b"MM":
        end = ">"
    else:
        raise ValueError("Not a TIFF (bad byte order mark)")
    magic, = struct.unpack(end + "H", raw[2:4])
    if magic != 42:
        raise ValueError(f"Not a classic TIFF (magic={magic})")
    ifd_off, = struct.unpack(end + "I", raw[4:8])

    n, = struct.unpack(end + "H", raw[ifd_off:ifd_off + 2])
    tags = {}
    for i in range(n):
        base = ifd_off + 2 + i * 12
        tag, typ, count = struct.unpack(end + "HHI", raw[base:base + 8])
        val_bytes = raw[base + 8:base + 12]
        if typ not in _TYPE:
            tags[tag] = (typ, count, val_bytes)
            continue
        code, size = _TYPE[typ]
        total = size * count
        if total <= 4:
            vals = struct.unpack(end + code * count, val_bytes[:total])
        else:
            ptr, = struct.unpack(end + "I", val_bytes)
            vals = struct.unpack(end + code * count, raw[ptr:ptr + total])
        tags[tag] = vals[0] if count == 1 else vals
    return end, tags


def _unpack_12bit(packed: np.ndarray, count: int) -> np.ndarray:
    """Unpack MSB-first packed 12-bit samples. 2 samples per 3 bytes."""
    n_pairs = (count + 1) // 2
    needed = n_pairs * 3
    if packed.size < needed:
        packed = np.concatenate([packed, np.zeros(needed - packed.size, np.uint8)])
    b = packed[:needed].reshape(n_pairs, 3).astype(np.uint16)
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
    s0 = (b0 << 4) | (b1 >> 4)
    s1 = ((b1 & 0x0F) << 8) | b2
    out = np.empty(n_pairs * 2, np.uint16)
    out[0::2] = s0
    out[1::2] = s1
    return out[:count]

def read_packed_12bit_tiff(path: Path | str) -> np.ndarray:
    """Return a (H, W) uint16 array of 12-bit codes in [0, 4095]."""
    raw = Path(path).read_bytes()
    end, tags = _read_ifd(raw)

    width = tags[0x0100]
    length = tags[0x0101]
    bits = tags[0x0102]
    compression = tags.get(0x0103, 1)
    spp = tags.get(0x0115, 1)
    strip_off = tags[0x0111]
    strip_bytes = tags[0x0117]

    if compression != 1:
        raise ValueError(f"Only uncompressed TIFFs supported (Compression={compression})")
    if spp != 1 or (isinstance(bits, tuple)):
        raise ValueError("Only single-channel (MONO) supported")
    if bits != 12:
        raise ValueError(f"This reader is for BitsPerSample=12, got {bits}")

    # StripOffsets/StripByteCounts may be scalars (single strip) or tuples.
    if isinstance(strip_off, tuple):
        chunks = [raw[o:o + c] for o, c in zip(strip_off, strip_bytes)]
        packed = np.frombuffer(b"".join(chunks), dtype=np.uint8)
    else:
        packed = np.frombuffer(raw[strip_off:strip_off + strip_bytes], dtype=np.uint8)

    samples = _unpack_12bit(packed, width * length)
    return samples.reshape(length, width)

if __name__ == "__main__":
    import sys
    truth = np.load("/home/user/workspace/test_packed12_truth.npy")
    got = read_packed_12bit_tiff("/home/user/workspace/test_packed12.tiff")
    print("shape:", got.shape, "dtype:", got.dtype)
    print("min/max:", got.min(), got.max())
    print("EXACT MATCH vs truth:", np.array_equal(got, truth))