from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
import scipy.fftpack
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy


def speckle_pattern_statistics(image: np.ndarray, bit_depth: int) -> dict[str, float]:
    """A function to perform diagnostics on the speckle pattern image.

    Parameters
    ----------
    image : np.ndarray
        2D numpy array representing the speckle pattern
    bit_depth : int
        Bit depth of the image (8 or 16)

    Returns
    -------
    dict
        Dictionary with diagnostic results
    """
    
    assert bit_depth in [8, 16], "Bit depth should be either 8 or 16."
    dynamic_range: int = 2**bit_depth - 1

    HFWHM, HeSquared, H_fit_stats, VFWHM, VeSquared, V_fit_stats, popt_H, popt_V, h_profile, v_profile = speckle_size(image)
    avg_speckle_size_fwhm = np.mean([HFWHM, VFWHM])
    avg_speckle_size_e2 = np.mean([HeSquared, VeSquared])

    # Calculate black/white ratio
    black_pixels = np.sum(image < (dynamic_range // 2))
    white_pixels = np.sum(image >= (dynamic_range // 2))
    ratio = black_pixels / white_pixels

    # Calculate mean intensity gradient (simple finite difference)
    grad_y, grad_x = np.gradient(image)
    mean_gradient = np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2))
    std_dev = np.std(image)
    avg = np.mean(image)
    contrast = std_dev / avg
    skewness = skew(image.flatten())
    kurt = kurtosis(image.flatten())
    entropy = shannon_entropy(image)
    peak_to_mean = np.max(image) / np.mean(image)

    results = {
        "black_white_ratio": ratio,
        "mean_intensity_gradient": mean_gradient,
        "std_dev_irradiance": std_dev,
        "avg_irradiance": avg,
        "contrast": contrast,
        "skewness": skewness,
        "kurtosis": kurt,
        "shannon_entropy": entropy,
        "peak_to_mean_ratio": peak_to_mean,
        "avg_speckle_size_fwhm": avg_speckle_size_fwhm,
        "avg_speckle_size_e2": avg_speckle_size_e2,
        "H_fit_stats": H_fit_stats,
        "V_fit_stats": V_fit_stats
    }

    return results

def speckle_pattern_plots(image: np.ndarray, bit_depth: int,  
                                save_path: Path | None = None, image_format: str = 'jpg') -> dict[str, Figure | Axes]:
    """A function to generate and save diagnostic plots for the speckle pattern.

    Parameters
    ----------
    image : np.ndarray
        2D numpy array representing the speckle pattern
    bit_depth : int
        Bit depth of the image (8 or 16)
    save_path : str (optional)
        Path to save the generated plots using example formatting to the folder if provided

    Returns
    -------
    dict
        Dictionary containing figures and axes of the generated plots
    """
    
    assert bit_depth in [8, 16], "Bit depth should be either 8 or 16."
    dynamic_range: int = 2**bit_depth - 1

    HFWHM, HeSquared, H_fit_stats, VFWHM, VeSquared, V_fit_stats, popt_H, popt_V, h_profile, v_profile = speckle_size(image)
    plots = {}

    # Set Seaborn theme
    sns.set_theme(style="darkgrid")
    matplotlib.rcParams['font.family'] = 'Sans-serif'
    plt.rc('text', usetex=True)
    fontsize1= 17

    fig, axes = plt.subplots(1, 1, figsize=(7.0, 5.0))
    ax = axes
    ax.imshow(image, cmap='gray', vmin=0, vmax=dynamic_range)
    cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize1-2)
    cbar.set_label('Irradiance', fontsize=fontsize1-2)
    ax.tick_params(axis='both', which='major', labelsize=fontsize1-2)
    ax.set_xlabel("Position [pixel]", fontsize=fontsize1-2)
    ax.set_ylabel("Position [pixel]", fontsize=fontsize1-2)
    ax.set_title("Speckle pattern", fontsize=fontsize1)

    if save_path is not None:
        # plt.savefig(f"{save_path}/speckle_pattern.jpg", dpi=300, format='jpg', bbox_inches='tight')
        plt.savefig(f"{save_path}/speckle_pattern." + f'{image_format}', dpi=300, format=image_format, bbox_inches='tight')

    plots['speckle_pattern_fig'] = fig
    plots['speckle_pattern_ax'] = ax

    f_image = scipy.fftpack.fft2(image)
    f_image_shifted = scipy.fftpack.fftshift(f_image)
    magnitude_spectrum = np.abs(f_image_shifted)
    magnitude_spectrum_log = np.log1p(magnitude_spectrum)
    fig, axes = plt.subplots(1, 1, figsize=(7.0, 5.0))
    ax = axes
    ax.imshow(magnitude_spectrum_log, cmap='gray')
    cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize1-2)
    ax.tick_params(axis='both', which='major', labelsize=fontsize1-2)
    ax.set_title("Spatial frequency (log scale)", fontsize=fontsize1)
    ax.set_xlabel("Frequency [1/pixel]", fontsize=fontsize1-2)
    ax.set_ylabel("Frequency [1/pixel]", fontsize=fontsize1-2)

    if save_path is not None:
        plt.savefig(f"{save_path}/frequency_spectrum." + f'{image_format}', dpi=300, format=image_format, bbox_inches='tight')

    plots['frequency_spectrum_fig'] = fig
    plots['frequency_spectrum_ax'] = ax

    fig, axes = plt.subplots(1, 1, figsize=(7.0, 5.0))
    ax = axes
    ax.hist(image.ravel(), density=True, bins=int(dynamic_range/10), color='blue', alpha=0.7, log=True)
    ax.set_title("Histogram of irradiance values", fontsize=fontsize1)
    ax.set_xlabel("Pixel value", fontsize=fontsize1-2)
    ax.set_ylabel("Density (log scale)", fontsize=fontsize1-2)
    ax.tick_params(axis='both', which='major', labelsize=fontsize1-2)

    if save_path is not None:
        plt.savefig(f"{save_path}/pixel_value_histogram." + f'{image_format}', dpi=300, format=image_format, bbox_inches='tight')

    plots['pixel_value_histogram_fig'] = fig
    plots['pixel_value_histogram_ax'] = ax

    plt.figure(figsize=(7.0, 5.0))
    x_H = np.arange(1, h_profile.size + 1)
    x_V = np.arange(1, v_profile.size + 1)
    plt.subplot(2, 1, 1)
    plt.plot(x_H, h_profile, 'b-', label='Horisontal autocov.',
                linewidth=2)
    plt.plot(x_H, gaussian(x_H, *popt_H), 'r--', label='Gaussian interpol.', linewidth=2)
    plt.title('Horisontal autocovariance', fontsize=fontsize1)
    plt.xlabel('Lag [pixels]', fontsize=fontsize1-2)
    plt.ylabel('Autocov. ' + r"[pixel$^2$]", fontsize=fontsize1-2)
    plt.legend(fontsize=fontsize1-2)
    plt.tick_params(axis='both', which='major', labelsize=fontsize1-2)
    plt.subplot(2, 1, 2)
    plt.plot(x_V, v_profile, 'b-', label='Vertical autocov.',
                linewidth=2)
    plt.plot(x_V, gaussian(x_V, *popt_V), 'r--', label='Gaussian interpol.', linewidth=2)
    plt.title('Vertical autocovariance', fontsize=fontsize1)
    plt.xlabel('Lag [pixels]', fontsize=fontsize1-  2)
    plt.ylabel('Autocov. ' + r"[pixel$^2$]", fontsize=fontsize1-2)
    plt.legend(fontsize=fontsize1-2)
    plt.tick_params(axis='both', which='major', labelsize=fontsize1-2)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(f"{save_path}/autocovariance." + f'{image_format}', dpi=300, format=image_format, bbox_inches='tight')

    # Extract figure and axis
    fig = plt.gcf()
    ax = plt.gca()

    plots['autocovariance_fig'] = fig
    plots['autocovariance_ax'] = ax

    return plots

def speckle_size(image: np.ndarray) -> tuple:
    """ A function to calculate speckle size from the autocovariance of the speckle pattern.

    Parameters
    ----------
    image : np.ndarray
        2D numpy array representing the speckle pattern

    Returns
    -------
    tuple
        HFWHM : float
            Full width at half maximum for horisontal profile.
        HeSquared : float
            1/e^2 width for horisontal profile.
        H_fit_stats : dict
            R-squared goodness of fit for horisontal profile.
        VFWHM : float
            Full width at half maximum for vertical profile.
        VeSquared : float
            1/e^2 width for vertical profile.
        V_fit_stats : dict
            R-squared goodness of fit for vertical profile.
        popt_H : array
            Optimal parameters for horisontal Gaussian fit.
        popt_V : array
            Optimal parameters for vertical Gaussian fit.
        h_profile : np.ndarray
            Horisontal autocovariance profile.
        v_profile : np.ndarray
            Vertical autocovariance profile.    
    """    
    # image = (image - np.mean(image)) / np.std(image)
    image = (image - np.mean(image))
    
    # Autocorrelation (FFT analysis)
    f_image_norm = scipy.fftpack.fft2(image)
    power_spectrum = np.abs(f_image_norm)**2
    autocorr = scipy.fftpack.ifft2(power_spectrum).real
    autocorr = scipy.fftpack.fftshift(autocorr)
    autocorr /= np.max(autocorr)
    centre_y, centre_x = np.array(autocorr.shape) // 2
    h_profile = autocorr[centre_y, :]
    v_profile = autocorr[:, centre_x]

    HFWHM, HeSquared, H_fit_stats, VFWHM, VeSquared, V_fit_stats, popt_H, popt_V = fit_gaussian(h_profile, v_profile)

    return HFWHM, HeSquared, H_fit_stats, VFWHM, VeSquared, V_fit_stats, popt_H, popt_V, h_profile, v_profile

def gaussian(x: np.ndarray, a1: float, b1: float, c1: float) -> np.ndarray:
    """A function to model Gaussian distribution.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    a1 : float
        Peak amplitude (height)
    b1 : float
        Center position (mean)
    c1 : float
        Standard deviation (width, spread)

    Returns
    -------
    np.ndarray
        _description_
    """      
    return a1 * np.exp(-((x - b1)/c1) ** 2)

def fit_gaussian(H: np.ndarray, V: np.ndarray) -> tuple:
    """ Fit Gaussian functions to the horisontal and vertical autocovariance profiles.

    Parameters
    ----------
    H : np.ndarray
        Horisontal autocovariance profile.
    V : np.ndarray
        Vertical autocovariance profile.

    Returns
    -------
    tuple
        HFWHM : float
            Full width at half maximum for horisontal profile.
        HeSquared : float
            1/e^2 width for horisontal profile.
        H_fit_stats : dict
            R-squared goodness of fit for horisontal profile.
        VFWHM : float
            Full width at half maximum for vertical profile.
        VeSquared : float
            1/e^2 width for vertical profile.
        V_fit_stats : dict
            R-squared goodness of fit for vertical profile.
        popt_H : array
            Optimal parameters for horisontal Gaussian fit.
        popt_V : array
            Optimal parameters for vertical Gaussian fit.
    """    
     
    range_H = np.arange(1, H.size + 1)
    range_V = np.arange(1, V.size + 1)

    low_H = np.where(H > 0.2)[0]
    low_V = np.where(V > 0.2)[0]

    popt_H, _ = curve_fit(gaussian, range_H[low_H], H[low_H], p0=[1, np.argmax(H), 1])
    popt_V, _ = curve_fit(gaussian, range_V[low_V], V[low_V], p0=[1, np.argmax(V), 1])

    # Extract FWHM and 1/e^2 widths
    HFWHM = 2 * popt_H[2] * np.sqrt(-np.log(0.5 / popt_H[0]))  # FWHM for horisontal
    VFWHM = 2 * popt_V[2] * np.sqrt(-np.log(0.5 / popt_V[0]))  # FWHM for vertical
    HeSquared = popt_H[2] * np.sqrt(-np.log(0.1353353 / popt_H[0]))  # 1/e^2 for horisontal
    VeSquared = popt_V[2] * np.sqrt(-np.log(0.1353353 / popt_V[0]))  # 1/e^2 for vertical

    # R-squared goodness of fit
    H_fit_stats = {'R_squared': 1 - np.sum((H[low_H] - gaussian(range_H[low_H], *popt_H)) ** 2) / np.sum((H[low_H] - np.mean(H[low_H])) ** 2)}
    V_fit_stats = {'R_squared': 1 - np.sum((V[low_V] - gaussian(range_V[low_V], *popt_V)) ** 2) / np.sum((V[low_V] - np.mean(V[low_V])) ** 2)}

    return HFWHM, HeSquared, H_fit_stats, VFWHM, VeSquared, V_fit_stats, popt_H, popt_V


