"""
Model Calibration Tools

Tools for displaying and recalibration model predictions.

Usage:
import mct

# get your model predictions and true class labels
preds, labels = ... 

# display calibration of original model
fig, estimate, ci = mct.display_calibration(preds,
                                            labels,
                                            bandwidth=0.05)
plt.show(block=False)

# Recalibrate predictions
calibrator = mct.create_calibrator(estimate.orig, estimate.calibrated)
calibrated = calibrator(preds)

# Display recalibrated predictions
plt.figure()
mct.display_calibration(calibrated, labels, bandwidth=0.05)
plt.show()
""" 

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from collections import namedtuple
from scipy import interpolate
from joblib import delayed, Parallel
from tqdm import tqdm

KdeResult = namedtuple('KdeResult',
                       'orig calibrated ici pos_intensity all_intensity')
"""Encapsulates the components of a KDE calibration.
    
    Args:
        orig (array-like of float in [0,1]): a grid of original probabilities, usually equally spaced over the domain.
        calibrated (array-like of float in [0,1]): the calibrated probabilities 
            corresponding to those in `orig`
        ici (float): the Integrated Calibration Index of `calibrated`, given 
            `all_intensity`
        pos_intensity (array-like of float): The intensity of elements with positive 
            labels, computed at values in `orig`
        all_intensity (array-like of float): The intensity of all elements, computed 
            at values in `orig`.
    """


def histograms(probs, actual, bins=100):
    """
    Calculates two histograms over [0, 1] by partitioning `probs` with `actual`
    and sorting each partition into `bins` sub-intervals.
    """
    actual = actual.astype(bool)
    edges, step = np.linspace(0., 1., bins, retstep=True, endpoint=False)
    idx = np.digitize(probs, edges) - 1
    top = np.bincount(idx, weights=actual, minlength=bins)
    bot = np.bincount(idx, weights=(~actual), minlength=bins)
    return top, bot, edges, step


def _compute_intensity(x_values, probs, kernel, bandwidth, **kde_args):
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, **kde_args)
    kde.fit(probs.reshape(-1, 1))
    log_density = kde.score_samples(x_values.reshape(-1, 1))

    # We want the area under `intensity` to be the number of samples
    intensity = np.exp(log_density) * len(probs)
    return intensity


def _compute_single_calibration(x_values, probs, actual, kernel, bandwidth,
                                **kde_args):
    positives = probs[actual == 1]
    pos_intensity = _compute_intensity(x_values, positives, kernel, bandwidth)
    all_intensity = _compute_intensity(x_values, probs, kernel, bandwidth,
                                       **kde_args)
    calibrated = pos_intensity / all_intensity
    ici = compute_ici(x_values, calibrated, all_intensity)
    return KdeResult(orig=x_values,
                     calibrated=calibrated,
                     ici=ici,
                     pos_intensity=pos_intensity,
                     all_intensity=all_intensity)
    
def _resample_calibration_wrapper(*args):
    cal = _compute_single_calibration(*args)
    return cal.calibrated, cal.ici, cal.pos_intensity, cal.all_intensity

def _resample_calibration(num_iterations, x_values, probs, actual, kernel,
                          bandwidth,n_jobs=1,**kde_args):
    calibrated,ici,pos_intensity,all_intensity = zip(*Parallel(n_jobs=n_jobs)(
        delayed(_resample_calibration_wrapper)(
            x_values,
            probs[idx], 
            actual[idx],
            kernel,
            bandwidth,
            **kde_args) for idx in tqdm([
            np.random.choice(probs.size, size=probs.size, replace=True) for _ in range(num_iterations)
            ])
        ))

    return KdeResult(orig=x_values,
                     calibrated=np.vstack(calibrated),
                     ici=ici,
                     pos_intensity=np.vstack(pos_intensity),
                     all_intensity=np.vstack(all_intensity))

def create_calibrator(orig, calibrated):
    """Create a function to calibrate new predictions.
    
    The calibration function is a linear interpolation of `calibrated` vs `orig`.
    Points outside the range of `orig` are interpolated as if (0,0) and (1,1)
    are included points.

    Args:
        orig (array-like of float): Original model predictions in [0,1]
        calibrated ([type]): Calibrated versions in [0,1] of `orig`.

    Returns:
        f(x) -> calibrated_x: a function returning calibrated versions 
            `calibrated_x` of inputs `x`, where x is array-like of float 
            in [0,1].
    """

    if orig[0] > 0:
        orig = np.insert(orig, 0, 0)
        calibrated = np.insert(calibrated, 0, 0)
    if orig[-1] < 1:
        orig = np.append(orig, 1.0)
        calibrated = np.append(calibrated, 1.0)
    return interpolate.interp1d(orig, calibrated, 'linear', bounds_error=True)


def compute_kde_calibration(probs,
                            actual,
                            resolution=0.01,
                            kernel='gaussian',
                            n_resamples=None,
                            bandwidth=0.1,
                            alpha=None,
                            pivot=True,
                            n_jobs=1,
                            **kde_args):
    """Generate a calibration curve using kernel density estimation.

    The curve is generated by computing the intensity (= probability density *
    number of samples) of the positive-labeled instances, and dividing that by
    the intensity of all instances.

    Uses bootstrap resampling to estimate the confidence intervals, if
    requested.

    Args:
        probs (array-like of float in [0,1]): model predicted probability for   
            each instance.
        actual (array-like of int in {0,1}): class label for each instance.
        resolution (float, optional): Desired curve grid resolution. 
            Defaults to 0.01.
        kernel (str, optional): Any valid kernel name for 
            sklearn.neighbors.KernelDensity. Defaults to 'gaussian'.
        n_resamples (int, optional): Number of iterations of bootstrap
            resampling for computing confidence intervals. If None (default),
            a value is chosen such that the CIs are reasonably repeatable. 
            Ignored if alpha=None.
        bandwidth (float, optional): Desired kernel bandwidth. Defaults to 0.1.
        alpha (float, optional): Desired significance level for the confidence 
            intervals. Defaults to None.
        pivot (bool, optional): Whether to use pivot-based confidence intervals (True: recommended),
            or empirical ones based on quantiles (False).
        **kde_args: Additional args for sklearn.neighbors.KernelDensity.

    Returns:
        A tuple containing:
            a KdeResult of the best estimates
            a KdeResult of the confidence intervals

    """

    x_min = max((0, np.amin(probs) - resolution))
    x_max = min((1, np.amax(probs) + resolution))

    x_values = np.arange(x_min, x_max, step=resolution)
    estimate = _compute_single_calibration(x_values,
                                           probs,
                                           actual,
                                           kernel=kernel,
                                           bandwidth=bandwidth,
                                           **kde_args)

    calibration_ci = None
    ici_ci = None
    pos_ci = None
    all_ci = None

    if alpha is not None:
        if n_resamples is None:
            # Choose a number of iterations such that there are about 50 points outside each end of the confidence interval.
            n_resamples = int(100 / alpha)
        samples = _resample_calibration(n_resamples, x_values, probs, actual,
                                        kernel, bandwidth,n_jobs=n_jobs, **kde_args)
        # Quantiles (empirical) confident intervals
        calibration_ci = np.quantile(samples.calibrated,
                                     (alpha / 2, 1 - alpha / 2),
                                     axis=0)
        ici_ci = np.quantile(samples.ici, (alpha / 2, 1 - alpha / 2))
        pos_ci = np.quantile(samples.pos_intensity, (alpha / 2, 1 - alpha / 2),
                             axis=0)
        all_ci = np.quantile(samples.all_intensity, (alpha / 2, 1 - alpha / 2),
                             axis=0)
        
    # Pivot based confident intervals
    if pivot:
        l = np.array([[0,1],[1,0]]) # revert help matrix
        calibration_ci_pivot = 2*estimate.calibrated - l@calibration_ci
        ici_ci_pivot = np.minimum(np.maximum(2*estimate.ici - l@ici_ci, 0),1) # Ensure ici is in [0,1]
        pos_ci_pivot = 2*estimate.pos_intensity - l@pos_ci
        all_ci_pivot = 2*estimate.all_intensity - l@all_ci
        
        ci = KdeResult(orig=x_values,
                       calibrated=calibration_ci_pivot,
                       ici=ici_ci_pivot,
                       pos_intensity=pos_ci_pivot,
                       all_intensity=all_ci_pivot)
    else:
        ci = KdeResult(orig=x_values,
                    calibrated=calibration_ci,
                    ici=ici_ci,
                    pos_intensity=pos_ci,
                    all_intensity=all_ci)
        
    return (estimate, ci)


def compute_ici(orig, calibrated, all_intensity):
    ici = (np.sum(all_intensity * np.abs(calibrated - orig)) /
           np.sum(all_intensity))
    return ici


def plot_histograms(top, bot, edges, resolution, *, ax=None):
    """
    Plots the two histograms generated by ``histograms``; the histogram of
    actual negatives is plotted underneath the x axis while the histogram of
    actual positives is plotted above.
    """
    if ax is None:
        ax = plt.gca()

    ax.hlines(y=0,
              xmin=0,
              xmax=1,
              linestyle='dashed',
              color='black',
              alpha=0.2)
    ax.bar(edges, top, width=resolution, color='C2') #positives
    ax.bar(edges, -bot, width=resolution, color='C3') # negatives
    # Set some sensible defaults - these can be overridden after the fact,
    # since we return the axes object
    ax.set_xlim((-0.05, 1.05))
    ax.set_xlabel('Predicted Probability')
    height = max(abs(x) for x in ax.get_ylim())
    ax.set_ylim((-height, height))
    ax.set_ylabel('Count')
    return ax


def plot_calibration_curve(orig,
                           calibrated,
                           calibrated_ci=None,
                           ici=None,
                           ici_ci=None,
                           pos_intensity=None,
                           all_intensity=None,
                           *,
                           label=None,
                           ax=None):
    """
    Plots a calibration curve.
    """
    plot_intensities = pos_intensity is not None and all_intensity is not None

    if ax is None:
        ax = plt.gca()

    ax.set_aspect('equal')
    limits = (-0.05, 1.05)
    ax.set_ylim(limits)
    ax.set_xlim(limits)

    ici_ci_label = ('' if ici_ci is None else
                    f' (ICI [{ici_ci[0]:0.3f}, {ici_ci[1]:0.3f}])')
    ici_label = '' if ici is None else f' (ICI {ici:0.3f})'

    ax.plot((0, 1), (0, 1), 'black', linewidth=0.2)
    ax.plot(orig, calibrated, label=f'Estimated Calibration{ici_label}')

    if calibrated_ci is not None:
        ax.fill_between(orig,
                        calibrated_ci[0],
                        calibrated_ci[1],
                        color='C0',
                        alpha=0.3,
                        edgecolor='C0',
                        label=f'Confidence Interval{ici_ci_label}')

    if plot_intensities:
        # We normalize the intensities to a max of 1, so they can plot on the same y axis as the calibration curve.
        pos_intensity /= all_intensity.max()
        all_intensity /= all_intensity.max()

        ax.plot(orig,
                pos_intensity,
                color='C1',
                alpha=0.4,
                label='Positive Intensity')

        ax.plot(orig,
                all_intensity,
                color='C2',
                alpha=0.4,
                label='All Intensity')

    ax.legend(loc='best')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Probability')

    if label is not None:
        ax.set_title(f'{label}')

    return ax


def display_calibration(probs,
                        actual,
                        *,
                        figure=None,
                        bins=100,
                        label=None,
                        show_ici=True,
                        alpha=0.05,
                        pivot=True,
                        n_resamples=None,
                        kernel='gaussian',
                        bandwidth=0.1,
                        plot_intensities=False,
                        n_jobs=1):
    """Generates a calibration display.
    
    The display contains by default a calibration curve with confidence intervals, an 
    estimate of the Integrated Calibration Index (ICI), and a histogram of 
    the positive and negative values.

    Args:
        See `compute_kde_calibration` for `probs`, `actual`, `kernel`, 
            `alpha`, `n_resamples`, `bandwidth`, and `plot_intensities`. 
        
        Args specific to this function are:
        figure (Matplotlib figure, optional): Figure to use for plotting. 
            If None (default) a new figure is created.
        bins (int, optional): Number of bins for value histograms. Defaults to 100.
        label (string, optional): Legend label for calibration curve. 
            Defaults to None.
        show_ici (bool, optional): If true (default), the ICI value is stated 
            in the legend. 

    Returns:
        (figure, KdeResult, KdeResult): A tuple of the figure object, the
            KDE estimate for the calibration curve, and the KDE estimate
            for the confidence intervals.
    """

    resolution = 1.0 / bins

    if figure is None:
        figure = plt.gcf()
    ax1, ax2 = figure.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw=dict(height_ratios=(3, 1)),
    )
    estimate, ci = compute_kde_calibration(probs,
                                           actual,
                                           resolution=resolution,
                                           kernel=kernel,
                                           bandwidth=bandwidth,
                                           alpha=alpha,
                                           pivot=pivot,
                                           n_resamples=n_resamples,
                                           n_jobs=n_jobs
                                           )

    ax1 = plot_calibration_curve(
        orig=estimate.orig,
        calibrated=estimate.calibrated,
        calibrated_ci=ci.calibrated,
        ici=estimate.ici if show_ici else None,
        ici_ci=ci.ici if show_ici else None,
        pos_intensity=estimate.pos_intensity if plot_intensities else None,
        all_intensity=estimate.all_intensity if plot_intensities else None,
        label=label,
        ax=ax1)
    ax1.set_xlabel('')
    ax2 = plot_histograms(*histograms(probs, actual, bins=bins), ax=ax2)
    ax2.set_box_aspect(1. / 3.)
    ax1.xaxis.set_ticks_position('none')
    figure.subplots_adjust(hspace=0)
    figure.tight_layout()
    return figure, estimate, ci
