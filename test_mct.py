"""
Tests for the model calibration & visualization tools.
"""
import numpy as np
import pytest

import mct


bool_mask = np.arange(10) % 2 == 0
ints_mask = bool_mask.astype(np.int)


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
