.. _sec_overview:

Overview
========

What is the DaCapo Toolbox?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DaCapo Toolbox contains a set of convenient tools for training neural networks on large microscopy datasets.
Specifically it focuses on providing a simple interface to many tools and techniques with an emphasis on
common interfaces such as torch/numpy/dask and other common scientific computing packages.
The main exception to this is the `funlib.persistence.Array` object. A thin wrapper around a `dask` array with
added metadata necessary for reliable training on diverse microscopy datasets.
