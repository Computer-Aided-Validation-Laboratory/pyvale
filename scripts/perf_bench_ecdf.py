"""Compare standalone NumPy ECDF construction with SciPy.

Validation uses SciPy. The earlier PyVale benchmark also used SciPy internally:
250 × 400 took 29.91 ms through PyVale versus 28.89 ms in the direct SciPy loop.
For 256 camera pixels with 32 × 128 samples each, it took 0.823 s versus 0.729 s.

The standalone NumPy trial below sorts finite, equal-length samples and returns
arrays directly. In a seven-repeat run it took 0.62 ms versus 28.62 ms for
250 × 400, and 3.0 ms versus 748 ms for the 256-pixel camera block. These are
construction timings with prepared inputs. The script estimates full-frame
camera times by scaling the measured block.
"""

import argparse
from collections.abc import Callable
from functools import partial
from statistics import median
from timeit import repeat

import numpy as np
from scipy import stats


def calculate_numpy_batch(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align finite, equal-length samples on the union of ECDF breakpoints."""
    quants = np.sort(samples, axis=-1)
    count = quants.shape[-1]
    ranks = np.arange(1, count + 1) / count
    # A tied value contributes one jump at its final rank.
    jump_ranks = np.empty(count, dtype=bool)
    if count > 1:
        changed = quants[..., :-1] != quants[..., 1:]
        jump_ranks[:-1] = np.any(changed.reshape(-1, count - 1), axis=0)
    jump_ranks[-1] = True
    if np.all(jump_ranks):
        return ranks, quants
    return ranks[jump_ranks], quants[..., jump_ranks]


def calculate_scipy_single(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cdf = stats.ecdf(samples).cdf
    return cdf.probabilities, cdf.quantiles


def calculate_scipy_batch(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build one SciPy ECDF per row and align the curves on a shared grid."""
    rows = samples.reshape(-1, samples.shape[-1])
    cdfs = [stats.ecdf(row).cdf for row in rows]
    probs = np.unique(np.concatenate([cdf.probabilities for cdf in cdfs]))
    quants = np.stack(
        [
            cdf.quantiles[np.searchsorted(cdf.probabilities, probs, side="left")]
            for cdf in cdfs
        ]
    )
    return probs, quants.reshape((*samples.shape[:-1], probs.size))


def assert_matches_reference(
    calculate_numpy: Callable[[], tuple[np.ndarray, np.ndarray]],
    calculate_scipy: Callable[[], tuple[np.ndarray, np.ndarray]],
) -> None:
    probs, quants = calculate_numpy()
    reference_probs, reference_quants = calculate_scipy()
    np.testing.assert_array_equal(probs, reference_probs)
    np.testing.assert_array_equal(quants, reference_quants)


def report_timings(
    samples: int,
    calculate_numpy: Callable[[], object],
    calculate_scipy: Callable[[], object],
    *,
    repeats: int,
) -> None:
    numpy_ms = 1_000 * median(repeat(calculate_numpy, number=1, repeat=repeats))
    scipy_ms = 1_000 * median(repeat(calculate_scipy, number=1, repeat=repeats))
    print(
        f"{samples:>10,d} {numpy_ms:>12.4f} {scipy_ms:>12.4f}"
        f" {scipy_ms / numpy_ms:>11.2f}x",
        flush=True,
    )


def calculate_scipy_camera(samples: np.ndarray) -> np.ndarray:
    """Run the reference loop over epistemic cases and grid alignment for each pixel."""
    views = np.empty_like(samples)
    for pixel, pixel_samples in enumerate(samples):
        _, views[pixel] = calculate_scipy_batch(pixel_samples)
    return views


def benchmark_workflows(
    rng: np.random.Generator, *, repeats: int, camera_pixels: int
) -> None:
    samples = rng.standard_normal((250, 400))
    calculate_view = partial(calculate_numpy_batch, samples)
    calculate_reference = partial(calculate_scipy_batch, samples)
    assert_matches_reference(calculate_view, calculate_reference)
    print("\nRealistic scalar workflow (250 epistemic × 400 aleatory)")
    print(f"{'Samples':>10} {'NumPy (ms)':>12} {'SciPy (ms)':>12} {'Speedup':>12}")
    report_timings(400, calculate_view, calculate_reference, repeats=repeats)

    frame_pixels = 640 * 480
    camera_samples = rng.standard_normal((camera_pixels, 32, 128))
    calculate_view_camera = partial(calculate_numpy_batch, camera_samples)
    calculate_reference_camera = partial(calculate_scipy_camera, camera_samples)
    probs, quants = calculate_view_camera()
    np.testing.assert_array_equal(probs, np.arange(1, 129) / 128)
    np.testing.assert_array_equal(quants, calculate_reference_camera())
    del quants
    numpy_seconds = median(repeat(calculate_view_camera, number=1, repeat=repeats))
    scipy_seconds = median(repeat(calculate_reference_camera, number=1, repeat=repeats))
    print("\nCamera workflow (640 × 480; 32 epistemic × 128 aleatory per pixel)")
    print(
        f"Measured {camera_pixels:,} pixels: NumPy {numpy_seconds:.3f} s,"
        f" SciPy {scipy_seconds:.3f} s ({scipy_seconds / numpy_seconds:.2f}x)."
    )
    if camera_pixels < frame_pixels:
        scale = frame_pixels / camera_pixels
        print(
            f"Full-frame serial estimate: NumPy {numpy_seconds * scale / 60:.1f} min,"
            f" SciPy {scipy_seconds * scale / 60:.1f} min."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-power", type=int, default=5)
    parser.add_argument("--max-power", type=int, default=16)
    parser.add_argument("--trans", type=int, default=8)
    parser.add_argument("--nuis", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--workflows-only", action="store_true")
    parser.add_argument(
        "--camera-pixels", type=int, default=256,
        help="Number of pixels to time from the 640 × 480 camera workload.",
    )
    args = parser.parse_args()
    if not 0 <= args.min_power <= args.max_power:
        parser.error("Require 0 <= min-power <= max-power.")
    if min(args.trans, args.nuis, args.repeats) < 1:
        parser.error("trans, nuis and repeats must be positive.")
    if not 1 <= args.camera_pixels <= 640 * 480:
        parser.error("camera-pixels must be between 1 and 307200.")

    sizes = [2**power for power in range(args.min_power, args.max_power + 1)]
    rng = np.random.default_rng(2751)
    heading = f"{'Samples':>10} {'NumPy (ms)':>12} {'SciPy (ms)':>12} {'Speedup':>12}"
    print(f"Median of {args.repeats} timings. Speedup = SciPy time / NumPy time.")
    if not args.workflows_only:
        benchmark_vector_lengths(sizes, rng, args.trans, args.nuis, args.repeats, heading)
    benchmark_workflows(
        np.random.default_rng(2751), repeats=args.repeats, camera_pixels=args.camera_pixels
    )


def benchmark_vector_lengths(
    sizes: list[int],
    rng: np.random.Generator,
    trans: int,
    nuis: int,
    repeats: int,
    heading: str,
) -> None:
    print(f"\nSingle 1D ECDF\n{heading}")
    for size in sizes:
        samples = rng.standard_normal(size)
        calculate_numpy = partial(calculate_numpy_batch, samples)
        calculate_scipy = partial(calculate_scipy_single, samples)
        assert_matches_reference(calculate_numpy, calculate_scipy)
        report_timings(
            size, calculate_numpy, calculate_scipy, repeats=repeats
        )

    print(f"\nBatched ECDF view ({trans} trans × {nuis} nuis)")
    print("One NumPy batch call vs. SciPy looping over cases and aligning the grids.")
    print(heading)
    for size in sizes:
        samples = rng.standard_normal((trans, nuis, size))
        calculate_view = partial(calculate_numpy_batch, samples)
        calculate_reference = partial(calculate_scipy_batch, samples)
        assert_matches_reference(calculate_view, calculate_reference)
        report_timings(
            size, calculate_view, calculate_reference, repeats=repeats
        )


if __name__ == "__main__":
    main()
