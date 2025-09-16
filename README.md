![daisy](docs/source/_static/dacapo-toolbox.png)
[![tests](https://github.com/pattonw/dacapo-toolbox/actions/workflows/tests.yaml/badge.svg)](https://github.com/pattonw/dacapo-toolbox/actions/workflows/tests.yaml)
[![ruff](https://github.com/pattonw/dacapo-toolbox/actions/workflows/ruff.yaml/badge.svg)](https://github.com/pattonw/dacapo-toolbox/actions/workflows/ruff.yaml)
[![ty](https://github.com/pattonw/dacapo-toolbox/actions/workflows/ty.yaml/badge.svg)](https://github.com/pattonw/dacapo-toolbox/actions/workflows/ty.yaml)
[![pypi](https://github.com/pattonw/dacapo-toolbox/actions/workflows/publish.yaml/badge.svg)](https://pypi.org/project/dacapo-toolbox/)
[![docs](https://github.com/pattonw/dacapo-toolbox/actions/workflows/docs.yaml/badge.svg)](https://github.com/pattonw/dacapo-toolbox/actions/workflows/docs.yaml)

# DaCapo-Toolbox: A set of tools for running ML experiments on large scale microscopy data

> A simple interface for large scale microscopy training and post processing

## What is DaCapo-Toolbox?

DaCapo-Toolbox wraps powerful tools for large scale microscopy training and evaluation such as:
[gunpowder](https://github.com/funkelab/gunpowder), [daisy](https://github.com/funkelab/daisy), and
[volara](https://github.com/e11bio/volara) in simple interfaces that abstract away the intricacies
of the specific tools and just provides interfaces in common formats such as
[zarr](https://zarr.readthedocs.io/en/stable/), [torch](https://pytorch.org), [dask](https://dask.org),
and [numpy](https://numpy.org).

## Installation

`pip install dacapo-toolbox`

## Usage

See the [DaCapo-Toolbox docs](https://pattonw.github.io/dacapo-toolbox)
