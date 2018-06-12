"""
Model Calibration Tools
"""
import numpy as np

def histograms(probs, actual, bins=100):
    """
    Calculates two histograms over [0, 1] by partitioning `probs` with `mask`
    and sorting each partition into `bins` sub-intervals.
    """
    actual = actual.astype(np.bool)
    edges, step = np.linspace(0., 1., bins, retstep=True, endpoint=False)
    idx = np.digitize(probs, edges) - 1
    top = np.bincount(idx, weights=actual, minlength=bins)
    bot = np.bincount(idx, weights=(~actual), minlength=bins)
    return top, bot, edges, step
