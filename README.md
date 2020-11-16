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

Suppose we have a binary classifier, some number *N* of instances, and that our
classifier is capable of generating probabilities associated with its
predictions.  For a given classifier, let `probs` be a 1-dim [array][ndArray]
(or array-like) of length *N* of the classifier's predicted probability *p* for
each input instance. Let `actual` be a 1-dim array-like of true class labels in
{0,1}.  Our classifier will be considered _well-calibrated_ if the fraction of
all instances with specific prediction *p* is close to *p*.  In other words, if
a perfectly calibrated classifier predicts *p* = 0.7 for a total 100 instances,
we will find that exactly 70 of those instances have label 1.

This module provides a few functions to aid in determining and visualizing how
well calibrated a classifier is.

```python
histograms(probs, actual, bins=100) -> (positives, negatives, edges, step)
```
`probs` and `actual` are defined as above, and `bins` specifies the number
of subintervals of `[0, 1]` to calculate.  The instances are grouped by
probability of prediction and then partitioned into two histograms by whether
or not they are actually included in the class.  The first histogram is
therefore all positives by predicted probability, and the second is all
negatives.  This functions also returns an array of the left `edges` of the
histogram and the histogram step-size (`step`), a real.  These values are
useful for plotting the histograms against each other.


```python kde_calibration_curve(probs, actual, resolution=0.01,
kernel='gaussian', bandwidth=0.1) -> (x_values, calibration, pos_intensity,
all_intensity) ``` `probs` and `actual` are defined as above.  The calibration
curve is computed on a set of points at equal `resolution` spacing over the
range of `probs`. The density of positive-labeled and all points is computed
using [KDE][kde], and the calibration curve is the ratio of those densities.
This functions return the domain points, the calibration, and the two densities
for plotting purposes.


```python
plot_histograms(top, bot, edges, step, *, ax=None) -> Axes
```
`ax` is assumed to be a [matplotlib Axes object][axes], if none is given,
we attempt to find the current one using [`pyplot.gca()`][gca].  We plot the
two histograms given by `histograms` on the given axes as a kind of hybrid
histogram; the positive class is plotted as a positive histogram, the negative
class as negative offsets from the y axis.  For examples please refer to the
ExampleUsage notebook.  The axes object is returned.


```python
plot_calibration_curve(x_values, calibration, pos_intensity=None, all_intensity=None, *, label=None, ax=None) -> Axes
```
 Plots the three curves vs. `x_values` on the given axes. For examples please refer to the ExampleUsage notebook.  The axes
object is returned. `ax` is as in `plot_histograms`.


```python
display_calibration(probs, actual, *, figure=None, bins=100, label=None, kernel='gaussian', bandwidth=0.1, ici=True, include_intensities=False) -> Figure
```
This function is a convenience function that runs the whole pipeline and
produces a [matplotlib Figure][figure].  If `figure` is not provided we attempt
to find one using [`pyplot.gcf()`][gcf].  If `include_intensities` is True,
intensities are plotted with the calibration curve.  If `ici` is true, the ICI is added to the legend. The
histograms are plotted below the calibration curves.  For examples please see
the accompanying ExampleUsage notebook.  The figure is returned.


[ndArray]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy-ndarray
[kde]: http://scikit-learn.org/stable/modules/density.html#kernel-density-estimation
[axes]: https://matplotlib.org/api/axes_api.html
[gca]: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.gca.html#matplotlib-pyplot-gca
[figure]: https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib-figure-figure
[gcf]: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.gcf.html#matplotlib-pyplot-gcf
