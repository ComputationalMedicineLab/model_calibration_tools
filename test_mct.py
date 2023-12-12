"""
Tests for the model calibration & visualization tools.
"""
import numpy as np
import pytest

import mct


bool_mask = np.arange(10) % 2 == 0
ints_mask = bool_mask.astype(int)


@pytest.mark.parametrize('mask',
                         (bool_mask, ints_mask),
                         ids=('bool', 'ints'))
def test_histogram_masking(mask):
    samples = np.linspace(0, 1, 10)

    exp_top = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    exp_bot = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
    exp_edges = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    exp_step = 0.1

    top, bot, edges, step = mct.histograms(samples, mask, bins=10)
    assert step == exp_step
    # They aren't _exactly_ equal b/c floating point errors
    # So we need to use np.allclose
    assert np.allclose(edges, exp_edges)
    assert np.allclose(top, exp_top)
    assert np.allclose(bot, exp_bot)
    
    
def toy_test_compute_kde_calibration():
    # Test pivot based ci
    # toy example
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.98, 0.55, 0.01, 0.05, 0.9])
    actual = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    estimate, ci_pivot = mct.compute_kde_calibration(probs, actual, pivot=True, n_resamples=10,alpha=0.05)
    _, ci_emp = mct.compute_kde_calibration(probs, actual, pivot=False, n_resamples=100, alpha=0.05)

if __name__ == '__main__':
    toy_test_compute_kde_calibration()