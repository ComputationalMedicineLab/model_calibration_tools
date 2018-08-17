"""
Model Calibration Tools
"""
from functools import wraps

import numpy as np
from sklearn.neighbors import KernelDensity


def histograms(probs, actual, bins=100):
    """
    Calculates two histograms over [0, 1] by partitioning `probs` with `actual`
    and sorting each partition into `bins` sub-intervals.
    """
    actual = actual.astype(np.bool)
    edges, step = np.linspace(0., 1., bins, retstep=True, endpoint=False)
    idx = np.digitize(probs, edges) - 1
    top = np.bincount(idx, weights=actual, minlength=bins)
    bot = np.bincount(idx, weights=(~actual), minlength=bins)
    return top, bot, edges, step


def kde_calibration_curve(probs, actual, bins=100,
                          kernel='gaussian',
                          bandwidth=0.1):
    """
    Generate a calibration curve smoothed via KDE.
    """
    x_axis = np.linspace(0, 1, bins)
    # Calculate the curve for actual values
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    train = probs[actual]
    kde.fit(train.reshape(-1, 1))
    # `scores` has been normalized so that the area under np.exp(scores) is
    # equal to one, but since we're using this to generate another curve, we
    # need to de-normalize the curve
    scores = kde.score_samples(x_axis.reshape(-1, 1))
    y_true = np.exp(scores) * len(train)

    # Calculate the curve for _all_ values
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kde.fit(probs.reshape(-1, 1))
    scores = kde.score_samples(x_axis.reshape(-1, 1))
    # See above note about scores
    y_total = np.exp(scores) * len(probs)

    # Now the Y values are true into total
    return (y_true / y_total, y_true, y_total)


def requires_matplotlib(func):
    """
    A decorator that handles importing matplotlib for the charting functions
    that need it.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
        except ImportError:
            return
        else:
            return func(*args, **kwargs)
    return wrapper


@requires_matplotlib
def plot_histograms(top, bot, edges, step, *, ax=None):
    """
    Plots the two histograms generated by ``histograms``; the falsey histogram
    is plotted underneath the x axis while the truthy histogram is plotted
    above.
    """
    if ax is None:
        ax = plt.gca()

    ax.hlines(y=0, xmin=0, xmax=1, linestyle='dashed', color='black', alpha=0.2)
    ax.bar(edges, top, width=step)
    ax.bar(edges, -bot, width=step)
    # Set some sensible defaults - these can be overridden after the fact,
    # since we return the axes object
    ax.set_xlim((-0.05, 1.05))
    ax.set_xlabel('Probability')
    height = max(abs(x) for x in ax.get_ylim())
    ax.set_ylim((-height, height))
    ax.set_ylabel('Instances')
    return ax


@requires_matplotlib
def plot_kde_calibration_curve(curve,
                               y_true=None,
                               y_total=None,
                               *,
                               label=None,
                               ax=None,
                               bins=100):
    """
    Plots a KDE smoothed calibration curve.
    """
    x_axis = np.linspace(0, 1, bins)
    inc_sources = y_true is not None and y_total is not None

    if ax is None:
        ax = plt.gca()

    ax.set_aspect('equal')
    limits = (-0.05, 1.05)
    ax.set_ylim(limits)
    ax.set_xlim(limits)

    lines, unit = [], (0, 1)
    lines.extend(ax.plot(unit, unit, 'k:', label='Perfectly Calibrated'))
    lines.extend(ax.plot(x_axis, curve, label='True / Total'))

    if inc_sources:
        # We normalize the sources to the interval 0, 1 so the plots will align
        # perfectly
        y_true  /= y_total.max()
        y_total /= y_total.max()
        lines.extend(ax.plot(x_axis, y_true, 'C1', alpha=0.4, label='True'))
        lines.extend(ax.plot(x_axis, y_total, 'C2', alpha=0.4, label='Total'))
    ax.legend(lines, [li.get_label() for li in lines], loc='best')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Percentage Actual')
    if label is not None:
        ax.set_title(f'{label}')
    return ax


@requires_matplotlib
def display_calibration(probs, actual,
                        *,
                        figure=None,
                        bins=100,
                        label=None,
                        kernel='gaussian',
                        bandwidth=0.1,
                        include_source_curves=False):
    """
    Generates and returns a matplotlib figure with two axes containing the
    model's kde-smoothed calibration curve and histograms.
    """
    if figure is None:
        figure = plt.gcf()
    ax1, ax2 = figure.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw=dict(height_ratios=(3, 1)),
    )
    curve, y_true, y_total = kde_calibration_curve(
        probs,
        actual,
        bins=bins,
        kernel=kernel,
        bandwidth=bandwidth
    )
    ax1 = plot_kde_calibration_curve(
        curve,
        y_true=y_true if include_source_curves else None,
        y_total=y_total if include_source_curves else None,
        label=label,
        ax=ax1,
        bins=bins,
    )
    ax1.set_xlabel('')
    ax2 = plot_histograms(
        *histograms(probs, actual, bins=bins),
        ax=ax2
    )
    ax1.xaxis.set_ticks_position('none')
    figure.tight_layout()
    return figure
