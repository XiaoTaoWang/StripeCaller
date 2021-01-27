StripeCaller
============
A Python implementation of the architectural stripe detecting algorithm described by Vian L et al [1]_.

Currently, this package contains 3 scripts: *call-stripes*, *pileup-stripes* and *stripe-plot*:

- call-stripes
  
  Call stripes from contact matrix. The only required input is the `cool <https://github.com/mirnylab/cooler>`_
  URI at certain resolution. The algorithm will detect horizontal (3') and vertical (5') stripes separately by
  searching for consecutive pixels with contact signals significantly higher than the local backgrounds. The output
  is a single file in bedpe format.

- pileup-stripes

  This script performs a pileup analysis on stripes.

- stripe-plot

  Visualize stripes on heatmap.


Installation
============
*stripecaller* is developed and tested on UNIX-like operating system, and following packages or softwares are
required, which can be installed through `conda <https://conda.io/miniconda.html>`_

a) Python 3.5+
b) Multiprocess
c) Numpy
d) Scipy
e) Matplotlib
f) Cooler

First, add some additional channels::

    $ conda config --add channels defaults
    $ conda config --add channels bioconda
    $ conda config --add channels conda-forge

Then execute the following command::

    $ conda install multiprocess numpy scipy matplotlib cooler

Finally, install the *stripecaller* by *pip*::

    $ pip install stripecaller

Usage
=====
Open a terminal, type ``call-stripes -h`` for help information.


Reference
=========
.. [1] Vian, L. et al. The Energetics and Physiological Impact of Cohesin Extrusion. Cell 175, 292-294, doi:10.1016/j.cell.2018.09.002 (2018).
