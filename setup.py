#! /usr/bin/env python
import io
from os.path import abspath, dirname, join
from setuptools import setup

__version__ = '0.1'

here = abspath(dirname(__file__))

with io.open(join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

setup(
    name='mct',
    version=__version__,
    description='Model Calibration Tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='John Still',
    author_email='john.m.still@vumc.org',
    python_requires='>=3.6.0',
    url='https://github.com/ComputationalMedicineLab/model_calibration_tools',
    py_modules=['mct'],
    install_requires=[
        'numpy',
        'scikit-learn',
        # required by the sklearn module we're using, but not a hard
        # requirement for sklearn
        'scipy',
        # required for plotting / charting etc
        'matplotlib',
    ],
    include_package_data=True,
    license='BSD',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering',
    ],
)
