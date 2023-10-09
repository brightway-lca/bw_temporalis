# bw_temporalis

[![PyPI](https://img.shields.io/pypi/v/bw_temporalis.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/bw_temporalis.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/bw_temporalis)][pypi status]
[![License](https://img.shields.io/pypi/l/bw_temporalis)][license]

[![Read the documentation at https://bw_temporalis.readthedocs.io/](https://img.shields.io/readthedocs/bw_temporalis/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/brightway-lca/bw_temporalis/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/brightway-lca/bw_temporalis/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/bw_temporalis/
[read the docs]: https://bw_temporalis.readthedocs.io/
[tests]: https://github.com/brightway-lca/bw_temporalis/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/brightway-lca/bw_temporalis
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Installation

You can install _bw_temporalis_ via [pip] from [PyPI]:

```console
$ pip install bw_temporalis
```

Currently Python 3.11 installations are broken as [scikit-network](https://scikit-network.readthedocs.io/en/latest/?badge=latest) only has pre-built wheels for version 0.31 - but this changed the [API for shortest path](https://github.com/sknetwork-team/scikit-network/blob/master/HISTORY.rst#0310-2023-05-22), and `bw_graph_tools` has not yet been adapted](https://github.com/brightway-lca/bw_graph_tools/issues/13).

You can also install using conda in the [cmutel channel](https://anaconda.org/cmutel/bw_temporalis).

## Usage notes

See [the teaching repo `from-the-ground-up`](https://github.com/brightway-lca/from-the-ground-up/tree/main/temporal) for examples on how to use this library.

This library uses the *net amount* in the technosphere and biosphere matrix, so caution should be taken in cases where multiple edges with temporal dynamics, especially with different numerical signs, link the same nodes. In general, these edges should be split across multiple processes.

The temporal resolution should not be less than seconds, it will be rounded up to seconds in the `TemporalDistribution`.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [BSD 3 Clause license][license],
_bw_temporalis_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.


<!-- github-only -->

[command-line reference]: https://bw_temporalis.readthedocs.io/en/latest/usage.html
[license]: https://github.com/brightway-lca/bw_temporalis/blob/main/LICENSE
[contributor guide]: https://github.com/brightway-lca/bw_temporalis/blob/main/CONTRIBUTING.md
