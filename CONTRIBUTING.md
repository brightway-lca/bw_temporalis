# Contributor Guide

Thank you for your interest in improving this project.
This project is open-source under the [BSD 3 Clause license] and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- [Code of Conduct]
- [License]
- [Source Code]
- [Documentation]
- [Issue Tracker]

## How to report a bug

Report bugs on the [Issue Tracker].

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker].

## How to set up your development environment

Install the package with development requirements:

```console
$ pip install -e ".[dev]"
```

## How to build the documentation locally

Make sure you have installed the `dev` and `docs` extras of the package.

```console
$ pip install -e ".[dev,docs]"
```

Build the documentation providing the `docs` directory at the root of the project as the source
and specifying the output directory.

```console
# use docs as source and docs/_build as output
sphinx-build docs docs/_build
```

## How to test the project


1. Install the package with development requirements:

```console
$ pip install -e ".[testing]"
```

2. Run the full test suite:

```console
$ pytest
```


Unit tests are located in the _tests_ directory,
and are written using the [pytest] testing framework.

[pytest]: https://pytest.readthedocs.io/

## How to submit changes

Open a [pull request] to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The test suite must pass without errors and warnings.
- Include unit tests.
- If your changes add functionality, update the documentation accordingly.

To run linting and code formatting checks before committing your change, you can install pre-commit as a Git hook by running one of following commands, depending on your dependencies manager:

```console
# conda or mamba
$ conda install pre-commit
```

or

```
$ pip install pre-commit
```


It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

[pull request]: https://github.com/brightway-lca/bw_temporalis/pulls
[License]: https://opensource.org/licenses/BSD-3-Clause
[Source Code]: https://github.com/brightway-lca/bw_temporalis
[Documentation]: https://bw_temporalis.readthedocs.io/
[Issue Tracker]: https://github.com/brightway-lca/bw_temporalis/issues


<!-- github-only -->

[Code of Conduct]: CODE_OF_CONDUCT.md
