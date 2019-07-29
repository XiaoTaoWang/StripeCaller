StripeCaller
============

Installation
============
*stripecaller* is developed and tested on UNIX-like operating system, and following packages or softwares are
required, which can be installed through `conda <Miniconda installer <https://conda.io/miniconda.html>`_

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

    $ pip install stripecaller-0.1.0-py3-none-any.whl


Usage
=====
Open a terminal, type ``stripecaller -h`` for help information.