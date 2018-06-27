#! /usr/bin/env python
"""
Produces classifications of MNIST so we have something to develop calibration
tools against.  The classifications are saved as a pandas dataframe.

Usage:
    $ ./gen_data.py --jobs 60 --output results.pkl
"""
import argparse
import multiprocessing
import time

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def get_mnist(predicate=None, shuffle=True):
    """Load, split, and shuffle the data.

    Returns:
        A 4-tuple of (X_train, X_test, y_train, y_test)
    """
    mnist = fetch_mldata('MNIST original', data_home='.')
    X, y = mnist['data'], mnist['target']
    X_train, X_test = X[:60_000], X[60_000:]
    y_train, y_test = y[:60_000], y[60_000:]
    if predicate is not None:
        y_train = predicate(y_train)
        y_test = predicate(y_test)
    if shuffle:
        # Some models need shuffling
        indices = np.random.permutation(60_000)
        X_train = X_train[indices]
        y_train = y_train[indices]
    return X_train, X_test, y_train, y_test


def predict_proba(clf, X_test):
    """Produce the probability vector for clf on X_test"""
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob = clf.decision_function(X_test)
        prob = (prob - prob.min()) / (prob.max() - prob.min())
    return prob


def evens(vec):
    """Convert vec (an ndarray) into a mask"""
    return vec % 2 == 0


def classify(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    prob = predict_proba(clf, X_test)
    return pred, prob


if __name__ == '__main__':
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', type=int)
    parser.add_argument('-o', '--output',
                        default='sample_classification_results.pkl')
    args = parser.parse_args()
    # 60 is the number of allowable cores on inferno - this prevents us
    # completely tieing up inferno's resources by accident
    n_jobs = args.jobs or min(multiprocessing.cpu_count(), 60)
    print(f'Using up to {n_jobs} cores')
    print(f'Results will be saved at {args.output}')
    print()

    # We create an even detector so that we have a binary classifier to work
    # with that has a decent number of true actuals vs the total
    X_train, X_test, y_train, y_test = get_mnist(predicate=evens)

    # note: this last one (KNeighborsClassifier) takes a while
    classifiers = (LogisticRegression(C=1., solver='lbfgs', n_jobs=n_jobs),
                   GaussianNB(),
                   LinearSVC(),
                   RandomForestClassifier(n_jobs=n_jobs),
                   SGDClassifier(tol=None, max_iter=5, n_jobs=n_jobs),
                   KNeighborsClassifier(n_jobs=n_jobs))

    headers = []
    results = []

    for clf in classifiers:
        msg = 'Starting classification via {:30}'
        # By default, sys.stdout buffers output until a newline is encountered,
        # but we're doing all the work in between now and when that happens so
        # we have to manually flush the output
        print(msg.format(clf.__class__.__name__ + '... '), end='', flush=True)
        start = time.time()
        pred, prob = classify(clf, X_train, y_train, X_test)
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        print(f'Done [{elapsed}]')
        headers.append(clf.__class__.__name__)
        results.extend([pred, prob])

    print(f'Saving results to {args.output}... ', end='', flush=True)
    # Read the data into a dataframe and serialize it to disk
    rows = np.array(results).T
    columns = pd.MultiIndex.from_product(
        [headers, ['Prediction', 'Probability']],
        names=['Classifier', 'Method'],
    )
    df = pd.DataFrame(rows, columns=columns)
    df['actual'] = y_test
    df.to_pickle(args.output)
    print('Done')
