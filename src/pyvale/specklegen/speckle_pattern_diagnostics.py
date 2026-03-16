import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mpf
import matplotlib.axes._axes as mpa
import scipy.fftpack
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from pyvale.sensorsim.visualopts import PlotOptsGeneral, SpecklePatternOpts

@dataclass(slots=True)
class SpeckleDiagnostics:
    black_white_ratio: float
    mean_intensity_gradient: float
    std_dev_irradiance: float
    avg_irradiance: float
    contrast: float
    skewness_value: float
    kurtosis_value: float
    shannon_entropy_value: float
    peak_to_mean_ratio: float
    avg_speckle_size_fwhm: float
    avg_speckle_size_e2: float
    fit_stats_r_sq_horis: float
    fit_stats_r_sq_vert: float

@dataclass(slots=True)
class GaussPeakFit:
    full_width_at_half_max: float # Full width at half maximum
    fit_stats_r_sq: float # R-squared goodness of fit
    one_over_e_sq_width: float # 1/e^2 width
    gauss_fit_optim_params: np.ndarray # Optimal parameters for Gaussian fit

def speckle_pattern_statistics(image: np.ndarray, bit_depth: int) -> SpeckleDiagnostics:
    """A function to perform diagnostics on the speckle pattern image.

    Parameters
    ----------
    image : np.ndarray, shape=(num_px_y, num_px)
        2D numpy array representing the speckle pattern
    bit_depth : int
        Bit depth of the image (8 or 16)

    Returns
    -------
    SpeckleDiagnostics dataclass
        Diagnostic results
    """

    assert bit_depth in [8, 16], "Bit depth should be either 8 or 16."
    dynamic_range: int = 2**bit_depth - 1

    stats_horis, stats_vert, _, _ = speckle_size(image)
    avg_speckle_size_fwhm = np.mean([stats_horis.full_width_at_half_max, stats_vert.full_width_at_half_max])
    avg_speckle_size_e2 = np.mean([stats_horis.one_over_e_sq_width, stats_vert.one_over_e_sq_width])

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

    results = SpeckleDiagnostics(
                   black_white_ratio=ratio,
                   mean_intensity_gradient=mean_gradient,
                   std_dev_irradiance=std_dev,
                   avg_irradiance=avg,
                   contrast=contrast,
                   skewness_value=skewness,
                   kurtosis_value=kurt,
                   shannon_entropy_value=entropy,
                   peak_to_mean_ratio=peak_to_mean,
                   avg_speckle_size_fwhm=avg_speckle_size_fwhm,
                   avg_speckle_size_e2=avg_speckle_size_e2,
                   fit_stats_r_sq_horis=stats_horis.fit_stats_r_sq,
                   fit_stats_r_sq_vert=stats_vert.fit_stats_r_sq)

    return results

def speckle_pattern_plot(image: np.ndarray, bit_depth: int,
                                plot_opts: PlotOptsGeneral | None = None,
                                speckle_opts: SpecklePatternOpts | None = None) -> tuple[mpf.Figure,mpa.Axes]:
    """A function to generate the speckle pattern plot.

    Parameters
    ----------
    image : np.ndarray, shape=(num_px_y, num_px)
        2D numpy array representing the speckle pattern
    bit_depth : int
        Bit depth of the image (8 or 16)
    plot_opts : PlotOptsGeneral | None, optional
        Options for controlling characteristics of the plot including the size
        of the figure, line widths etc., by default None. If None a default
        plot options dataclass is created.
    speckle_opts : SpecklePatternOpts | None, optional
        Options for controlling characteristics of the speckle pattern plot including 
        x and y axes labels etc., by default None. If None a default
        plot options dataclass is created.
    
    Returns
    -------
    tuple[mpf.Figure,mpa.Axes]
        Figure and axes object for the matplotlib plot that is created.
    """
    
    assert bit_depth in [8, 16], "Bit depth should be either 8 or 16."
    dynamic_range: int = 2**bit_depth - 1

    if plot_opts is None:
        plot_opts = PlotOptsGeneral()
    if speckle_opts is None:
        speckle_opts = SpecklePatternOpts()

    #---------------------------------------------------------------------------
    # Figure canvas setup
    (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_landscape,
                             layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    #---------------------------------------------------------------------------
    # Plot speckle pattern
    ax.imshow(image, cmap=plot_opts.cmap_seq, vmin=0, vmax=dynamic_range)

    #---------------------------------------------------------------------------
    # Axis / legend labels and options
    ax.set_xlabel(speckle_opts.x_label,
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax.set_ylabel(speckle_opts.y_label,
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    cbar = plt.colorbar(ax.images[0], ax=ax)
    cbar.ax.tick_params(labelsize=plot_opts.font_tick_size)
    cbar.set_label(speckle_opts.cmap_title, fontsize=plot_opts.font_leg_size)
    ax.tick_params(axis='both', which='major', labelsize=plot_opts.font_tick_size)
    ax.set_title(speckle_opts.title, fontsize=plot_opts.font_head_size)

    return (fig, ax)

def frequency_spectrum_plot(image: np.ndarray,
                                plot_opts: PlotOptsGeneral | None = None,
                                speckle_opts: SpecklePatternOpts | None = None) -> tuple[mpf.Figure,mpa.Axes]:
    """A function to generate the frequency spectrum plot for the speckle pattern.

    Parameters
    ----------
    image : np.ndarray, shape=(num_px_y, num_px)
        2D numpy array representing the speckle pattern
    plot_opts : PlotOptsGeneral | None, optional
        Options for controlling characteristics of the plot including the size
        of the figure, line widths etc., by default None. If None a default
        plot options dataclass is created.
    speckle_opts : SpecklePatternOpts | None, optional
        Options for controlling characteristics of the frequency spectrum plot including 
        x and y axes labels etc., by default None. If None a default
        plot options dataclass is created.
    
    Returns
    -------
    tuple[mpf.Figure,mpa.Axes]
        Figure and axes object for the matplotlib plot that is created.
    """

    if plot_opts is None:
        plot_opts = PlotOptsGeneral()
    if speckle_opts is None:
        speckle_opts = SpecklePatternOpts()

    #---------------------------------------------------------------------------
    # Figure canvas setup
    (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_landscape,
                             layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    #---------------------------------------------------------------------------
    # Plot frequency spectrum
    f_image = scipy.fftpack.fft2(image)
    f_image_shifted = scipy.fftpack.fftshift(f_image)
    magnitude_spectrum = np.abs(f_image_shifted)
    magnitude_spectrum_log = np.log1p(magnitude_spectrum)
    ax.imshow(magnitude_spectrum_log, cmap=plot_opts.cmap_seq)

    #---------------------------------------------------------------------------
    # Axis / legend labels and options
    ax.set_xlabel(speckle_opts.x_label,
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax.set_ylabel(speckle_opts.y_label,
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    cbar = plt.colorbar(ax.images[0], ax=ax)
    cbar.ax.tick_params(labelsize=plot_opts.font_tick_size)
    cbar.set_label(speckle_opts.cmap_title, fontsize=plot_opts.font_leg_size)
    ax.tick_params(axis='both', which='major', labelsize=plot_opts.font_tick_size)
    ax.set_title(speckle_opts.title, fontsize=plot_opts.font_head_size)

    return (fig, ax)


def pixel_value_histogram_plot(image: np.ndarray, bit_depth: int,
                                plot_opts: PlotOptsGeneral | None = None,
                                speckle_opts: SpecklePatternOpts | None = None) -> tuple[mpf.Figure,mpa.Axes]:
    """A function to generate pixel value histogram plot for the speckle pattern.

    Parameters
    ----------
    image : np.ndarray, shape=(num_px_y, num_px)
        2D numpy array representing the speckle pattern
    bit_depth : int
        Bit depth of the image (8 or 16)
    plot_opts : PlotOptsGeneral | None, optional
        Options for controlling characteristics of the plot including the size
        of the figure, line widths etc., by default None. If None a default
        plot options dataclass is created.
    speckle_opts : SpecklePatternOpts | None, optional
        Options for controlling characteristics of the pixel value histogram plot including 
        x and y axes labels etc., by default None. If None a default
        plot options dataclass is created.
    
    Returns
    -------
    tuple[mpf.Figure,mpa.Axes]
        Figure and axes object for the matplotlib plot that is created.
    """
    
    assert bit_depth in [8, 16], "Bit depth should be either 8 or 16."
    dynamic_range: int = 2**bit_depth - 1

    if plot_opts is None:
        plot_opts = PlotOptsGeneral()
    if speckle_opts is None:
        speckle_opts = SpecklePatternOpts()

    #---------------------------------------------------------------------------
    # Figure canvas setup
    (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_landscape,
                             layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    #---------------------------------------------------------------------------
    # Plot pixel value histogram
    ax.hist(image.ravel(), density=True, bins=int(dynamic_range/10), log=True)

    #---------------------------------------------------------------------------
    # Axis / legend labels and options
    ax.set_xlabel(speckle_opts.x_label,
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax.set_ylabel(speckle_opts.y_label,
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax.tick_params(axis='both', which='major', labelsize=plot_opts.font_tick_size)
    ax.set_title(speckle_opts.title, fontsize=plot_opts.font_head_size)

    return (fig, ax)


def autocovariance_plot(image: np.ndarray,
                                plot_opts: PlotOptsGeneral | None = None,
                                speckle_opts: SpecklePatternOpts | None = None) -> tuple[mpf.Figure,mpa.Axes]:
    """A function to generate autocovariance plot for the speckle pattern.

    Parameters
    ----------
    image : np.ndarray, shape=(num_px_y, num_px)
        2D numpy array representing the speckle pattern
    bit_depth : int
        Bit depth of the image (8 or 16)
    plot_opts : PlotOptsGeneral | None, optional
        Options for controlling characteristics of the plot including the size
        of the figure, line widths etc., by default None. If None a default
        plot options dataclass is created.
    speckle_opts : SpecklePatternOpts | None, optional
        Options for controlling characteristics of the autocovariance plot including 
        x and y axes labels etc., by default None. If None a default
        plot options dataclass is created.
    
    Returns
    -------
    tuple[mpf.Figure,mpa.Axes]
        Figure and axes object for the matplotlib plot that is created.
    """

    if plot_opts is None:
        plot_opts = PlotOptsGeneral()
    if speckle_opts is None:
        speckle_opts = SpecklePatternOpts()

    stats_horis, stats_vert, h_profile, v_profile = speckle_size(image)
    x_H = np.arange(1, h_profile.size + 1)
    x_V = np.arange(1, v_profile.size + 1)

    #---------------------------------------------------------------------------
    # Figure canvas setup
    fig, axes = plt.subplots(2, 1,figsize=plot_opts.single_fig_size_landscape,
                             layout='constrained')
    fig.set_dpi(plot_opts.resolution)
    ax1 = axes[0]
    ax2 = axes[1]

    #---------------------------------------------------------------------------
    # Plot autocovariance

    # Top subplot (horisontal)

    ax1.plot(x_H, h_profile, 'b-', label='Horisontal autocov.', linewidth=2)
    ax1.plot(x_H, gaussian(x_H, *stats_horis.gauss_fit_optim_params), 
             'r--', label='Gaussian interpol.', linewidth=2)
    
    # Bottom subplot (vertical)

    ax2.plot(x_V, v_profile, 'b-', label='Vertical autocov.', linewidth=2)
    ax2.plot(x_V, gaussian(x_V, *stats_vert.gauss_fit_optim_params), 
             'r--', label='Gaussian interpol.', linewidth=2)

    #---------------------------------------------------------------------------
    # Axis / legend labels and options

    # Top subplot (horisontal)

    ax1.set_title('Horisontal', fontsize=plot_opts.font_head_size)
    ax1.set_xlabel(speckle_opts.x_label, fontsize=plot_opts.font_ax_size)
    ax1.set_ylabel(speckle_opts.y_label, fontsize=plot_opts.font_ax_size)
    ax1.legend(fontsize=plot_opts.font_leg_size)
    ax1.tick_params(axis='both', which='major',labelsize=plot_opts.font_tick_size)

    # Bottom subplot (vertical)

    ax2.set_title('Vertical',fontsize=plot_opts.font_head_size)
    ax2.set_xlabel(speckle_opts.x_label, fontsize=plot_opts.font_ax_size)
    ax2.set_ylabel(speckle_opts.y_label, fontsize=plot_opts.font_ax_size)
    ax2.legend(fontsize=plot_opts.font_leg_size)
    ax2.tick_params(axis='both', which='major', labelsize=plot_opts.font_tick_size)

    plt.suptitle(speckle_opts.title)

    return (fig, axes)


def speckle_size(image: np.ndarray) -> tuple[GaussPeakFit, GaussPeakFit, np.ndarray, np.ndarray]:
    """ A function to calculate speckle size from the autocovariance of the speckle pattern.

    Parameters
    ----------
    image : np.ndarray, shape=(num_px_y, num_px)
        2D numpy array representing the speckle pattern

    Returns
    -------
    tuple[GaussPeakFit, GaussPeakFit, np.ndarray, np.ndarray]
        stats_horis : GaussPeakFit dataclass
            Parameters related to fitting Gaussian function to the horisontal profile.
        stats_vert : GaussPeakFit dataclass
            Parameters related to fitting Gaussian function to the vertical profile.
        h_profile : np.ndarray, shape=(num_px_x,)
            Horisontal autocovariance profile.
        v_profile : np.ndarray, shape=(num_px_y,)
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

    stats_horis, stats_vert = fit_gaussian(h_profile, v_profile)

    return stats_horis, stats_vert, h_profile, v_profile

def gaussian(x: np.ndarray, a1: float, b1: float, c1: float) -> np.ndarray:
    """A function to model Gaussian distribution.

    Parameters
    ----------
    x : np.ndarray, any shape
        Input array of any size (element-wise operations).
    a1 : float
        Peak amplitude (height)
    b1 : float
        Center position (mean)
    c1 : float
        Standard deviation (width, spread)

    Returns
    -------
    np.ndarray
        Gaussian curve values at the locations of x array (same shape as input x array)
    """      
    return a1 * np.exp(-((x - b1)/c1) ** 2)

def fit_gaussian(H: np.ndarray, V: np.ndarray) -> tuple[GaussPeakFit, GaussPeakFit]:
    """ Fit Gaussian functions to the horisontal and vertical autocovariance profiles.

    Parameters
    ----------
    H : np.ndarray, shape=(num_px_x,)
        Horisontal autocovariance profile.
    V : np.ndarray, shape=(num_px_y,)
        Vertical autocovariance profile.

    Returns
    -------
    tuple[GaussPeakFit, GaussPeakFit]
        stats_horis : GaussPeakFit dataclass
            Parameters related to fitting Gaussian function to the horisontal profile.
        stats_vert : GaussPeakFit dataclass
            Parameters related to fitting Gaussian function to the vertical profile.
    """ 

    range_horis = np.arange(1, H.size + 1)
    range_vert = np.arange(1, V.size + 1)

    low_horis = np.where(H > 0.2)[0]
    low_vert = np.where(V > 0.2)[0]

    gauss_fit_optim_params_horis, _ = curve_fit(gaussian, range_horis[low_horis], H[low_horis], p0=[1, np.argmax(H), 1])
    gauss_fit_optim_params_vert, _ = curve_fit(gaussian, range_vert[low_vert], V[low_vert], p0=[1, np.argmax(V), 1])

    # Extract full width at half maximum (FWHM) and 1/e^2 widths

     # FWHM for horisontal
    full_width_at_half_max_horis_1 = 2 * gauss_fit_optim_params_horis[2]
    full_width_at_half_max_horis_2 = np.sqrt(-np.log(0.5 / gauss_fit_optim_params_horis[0]))
    full_width_at_half_max_horis = full_width_at_half_max_horis_1 * full_width_at_half_max_horis_2
    # FWHM for vertical
    full_width_at_half_max_vert_1 = 2 * gauss_fit_optim_params_vert[2]
    full_width_at_half_max_vert_2 = np.sqrt(-np.log(0.5 / gauss_fit_optim_params_vert[0]))
    full_width_at_half_max_vert = full_width_at_half_max_vert_1 * full_width_at_half_max_vert_2
    # 1/e^2 for horisontal
    one_over_e_sq_width_horis = gauss_fit_optim_params_horis[2] * np.sqrt(-np.log(0.1353353 / gauss_fit_optim_params_horis[0]))
    # 1/e^2 for vertical
    one_over_e_sq_width_vert = gauss_fit_optim_params_vert[2] * np.sqrt(-np.log(0.1353353 / gauss_fit_optim_params_vert[0]))

    # R-squared goodness of fit
    fit_stats_r_sq_horis_1 = np.sum((H[low_horis] - gaussian(range_horis[low_horis], *gauss_fit_optim_params_horis)) ** 2)
    fit_stats_r_sq_horis_2 = np.sum((H[low_horis] - np.mean(H[low_horis])) ** 2)
    fit_stats_r_sq_horis = 1 - fit_stats_r_sq_horis_1 / fit_stats_r_sq_horis_2

    fit_stats_r_sq_vert_1 = np.sum((V[low_vert] - gaussian(range_vert[low_vert], *gauss_fit_optim_params_vert)) ** 2)
    fit_stats_r_sq_vert_2 = np.sum((V[low_vert] - np.mean(V[low_vert])) ** 2)
    fit_stats_r_sq_vert = 1 - fit_stats_r_sq_vert_1 / fit_stats_r_sq_vert_2

    stats_horis = GaussPeakFit(
                  full_width_at_half_max=full_width_at_half_max_horis,
                  fit_stats_r_sq=fit_stats_r_sq_horis,
                  one_over_e_sq_width=one_over_e_sq_width_horis,
                  gauss_fit_optim_params=gauss_fit_optim_params_horis)
    
    stats_vert = GaussPeakFit(
                  full_width_at_half_max=full_width_at_half_max_vert,
                  fit_stats_r_sq=fit_stats_r_sq_vert,
                  one_over_e_sq_width=one_over_e_sq_width_vert,
                  gauss_fit_optim_params=gauss_fit_optim_params_vert)

    return (stats_horis, stats_vert)


