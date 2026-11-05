![dacapo-toolbox](docs/source/_static/dacapo-toolbox-logo.png)
[![tests](https://github.com/pattonw/dacapo-toolbox/actions/workflows/tests.yaml/badge.svg)](https://github.com/pattonw/dacapo-toolbox/actions/workflows/tests.yaml)
[![lint](https://github.com/pattonw/dacapo-toolbox/actions/workflows/lint.yaml/badge.svg)](https://github.com/pattonw/dacapo-toolbox/actions/workflows/lint.yaml)
[![docs](https://github.com/pattonw/dacapo-toolbox/actions/workflows/docs.yaml/badge.svg)](https://pattonw.github.io/dacapo-toolbox)
[![publish](https://github.com/pattonw/dacapo-toolbox/actions/workflows/publish.yaml/badge.svg)](https://pypi.org/project/dacapo-toolbox/)

# DaCapo-Toolbox

> A simple interface for large scale microscopy training and post processing

**Documentation:** https://pattonw.github.io/dacapo-toolbox

## What is DaCapo-Toolbox?

DaCapo-Toolbox wraps powerful tools for large scale microscopy training and evaluation such as:
[gunpowder](https://github.com/funkelab/gunpowder), [daisy](https://github.com/funkelab/daisy), and
[volara](https://github.com/e11bio/volara) in simple interfaces that abstract away the intricacies
of the specific tools and just provides interfaces in common formats such as
[zarr](https://zarr.readthedocs.io/en/stable/), [torch](https://pytorch.org), [dask](https://dask.org),
and [numpy](https://numpy.org).

## Install

```bash
pip install dacapo-toolbox    # or: uv add dacapo-toolbox
```

## Usage

See the [DaCapo-Toolbox docs](https://pattonw.github.io/dacapo-toolbox)

## Contributing

The commands below are the ones CI runs, so a green local run means a green PR.

```bash
git clone https://github.com/pattonw/dacapo-toolbox
cd dacapo-toolbox
uv sync --extra vis --extra postprocessing          # dev tools + the extras the tests use (from uv.lock)
uv run pytest                                       # tests with coverage
uv run pytest --no-cov tests/<file>.py              # one file, without the coverage floor
uv run ruff check . && uv run ruff format --check . # lint and formatting
uv run ty check src/dacapo_toolbox                  # types
just docs                                           # docs (needs just, pandoc, ffmpeg)
```

`uv.lock` pins the development environment; the `lock-upgrade` workflow refreshes it monthly and opens a pull request.
