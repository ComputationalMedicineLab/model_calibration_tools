Model Calibration Visualization Tools
====================================

Installation
------------

```shell
git clone https://github.com/ComputationalMedicineLab/model_calibration_tools.git
cd model_calibration_tools
pip install .  # use -e if you want to develop on the toolkit itself
```

Usage
-----

For examples, see the ExampleUsage notebook.

Suppose we have a binary classifier, some number *N* of instances, and that our classifier is capable of generating probabilities associated with its predictions.  For a given classifier, let `probs` be a 1-dim [array][ndArray] (or array-like) of length *N* of the classifier's predicted probability *p* for each input instance. Let `actual` be a 1-dim array-like of true class labels in {0,1}.  Our classifier will be considered _well-calibrated_ if the fraction of all instances with predicted probability *p* is actually close to *p*.  In other words, if a perfectly calibrated classifier predicts *p* = 0.7 for a total 100 instances, we will find that exactly 70 of those instances have label 1.

This module provides a few functions to visualize and improve the calibration of a classifier.

Calibration curve results are returned as a named tuple:

```python
KdeResult = namedtuple('KdeResult',
                       'orig calibrated ici pos_intensity all_intensity')
```

where the calibration curve is represented as `orig` vs. `calibrated`, `ici` is the Integrated Calibration Index (ICI) of the curve, which takes into account not only the errors of the predictions, but the distribution of those predictions, putting more weight on areas of the curve where there are more predictions. The arrays `pos_intensity` (the intensity of predictions for positive instances), and `all_intensity` (the intensity of predictions for all instances) are provided for troubleshooting purposes; the calibration curve is `pos_intensity / all_intensity`.
  
**Public functions** include:

```python
display_calibration(probs, actual, *, figure=None, bins=100, label=None, kernel='gaussian', 
    bandwidth=0.1, ici=True, plot_intensities=False) -> Figure, estimate, confidence_intervals
```

The top-level convenience function that runs the whole pipeline and produces a [matplotlib Figure][figure], returning the figure, the calibration `estimate`, and the `confidence_intervals`, with legend `label`.  If `figure` is not provided we attempt to find one using [`pyplot.gcf()`][gcf].  If `plot_intensities` is True, troubleshooting intensities are plotted with the calibration curve.  If `ici` is true, the ICI is added to the legend. The histograms are plotted below the calibration curves.  The figure is returned.

```python
create_calibrator(orig, calibrated) -> f(orig_new)
```

The main function for recalibrating predictions. Returns a function `f` that maps new original predictions `orig_new` onto the curve described by `orig`, `calibrated`, linearly interpolating between points. When extrapolation beyond the boundaries of `orig` is needed, `f` considers `(0,0)` an d`(1,1)` to be part of the calibration curve.

Functions that do various pieces of all this include the following:

```python
compute_kde_calibration(probs, actual, resolution=0.01, kernel='gaussian',
    n_resamples=None, bandwidth=0.1, alpha=None) -> (estimate, confidence_intervals)
```

Computes the calibration curve of `probs`, given the `actual` labels. The curve is computed on a set of points at equal `resolution` spacing over the range of `probs`. The Integrated Calibration Index for that curve is also computed. 

If `alpha` is not None, then bootstrap resampling over instances is used to create `(1-alpha)` confidence intervals of the curve, the ICI, and the intensities. 

Results are returned as `KdeResults` of the best curve `estimate` and the `confidence_intervals` (which is None if `alpha` is None).

```python
compute_ici(orig, calibrated, all_intensity) -> float
```

Computes the ICI of the curve given by `orig` vs `calibrated`, with prediction intensity given by `all_intensity`.

```python
plot_calibration_curve(orig, calibrated, calibrated_ci=None, ici=None, ici_ci=None, 
    pos_intensity=None, all_intensity=None, *, label=None, ax=None) -> Axes
```

 Plots `calibrated` vs. `orig` on the given axes `ax`, and states the ICI in the plot legend if given. If `calibrated_ci` is given, plots the confidence interval as well, and the ici_ci if given. If both intensities are given, then they are both plotted. The axes object is returned.

```python
plot_histograms(top, bot, edges, step, *, ax=None) -> Axes
```

Plots the positive and negative histograms, with positive instances above the axis and negatives below. `ax` is assumed to be a [matplotlib Axes object][axes], if none is given, we attempt to find the current one using [`pyplot.gca()`][gca].   The axes object is returned.

```python
histograms(probs, actual, bins=100) -> (positives, negatives, edges, step)
```

Computes the histograms of actually positive and actually negative instances. `probs` and `actual` are defined as above, and `bins` specifies the number of histogram bins to use.  Bin `edges` and the histogram (`step`) size for the histograms are also returned.  The histograms are useful in understanding where the predictions are concentrated.

[ndArray]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy-ndarray
[kde]: http://scikit-learn.org/stable/modules/density.html#kernel-density-estimation
[axes]: https://matplotlib.org/api/axes_api.html
[gca]: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.gca.html#matplotlib-pyplot-gca
[figure]: https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib-figure-figure
[gcf]: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.gcf.html#matplotlib-pyplot-gcf
