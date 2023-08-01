# Python New Project Package

This repository holds a minimal but complete `python3` package
with the following quality services already included:

 - [`setuptools`][100] flow for packaging;
 - [`unittest`][101] test suite sub-package suited for Test Driven Development;
 - [`coverage`][102] for test suite;
 - [`bandit`][111] and [`safety`][114] for security tests;
 - [`pylint`][103] and [`flake8`][112] for syntax checks;
 - [`mypy`][104] for type hints checks and errors;
 - [`black`][105] and [`isort`][106] for code formatting;
 - [`jupyter`][107] notebooks rendering (with Sphinx integration);
 - [`Sphinx`][108] documentation builder;
 - [`anybadge`][109] for any session badges;  
 - [`nox`][110] for session coordination;
 - GitLab CI Pipeline and GitHub Workflow (future).

## How to?

### Start a new project

To start a new project from this package few steps are required:

 1. Fork this [repository](https://github.com/jlandercy/newproject) to your hub
    account and rename it;
 2. Clone the forked repository to your workspace;
 3. Rename the project by renaming the package directory;
 4. Start to implement your project as usual.

### Check package capabilities

Running Code Quality tool suite for this package requires
Code Quality packages are installed (defined in [`./docs/requirements.txt`][201]):

```bash
python -m pip install -r ./docs/requirements_ci.txt
```

Code Quality environment is also available in a dedicated [Docker image][300].

Then, to check package capabilities (aka `nox` sessions), issue:

```bash
nox --list

- clean -> Package Code Cleaner
- package -> Package Builds (badge)
- install -> Package Installer
- uninstall -> Package Uninstaller
* tests -> Package Test Suite Report (badge)
* coverage -> Package Test Suite Coverage Report (badge)
* security -> Package Security Report (badges)
* linter -> Package Linter Report (badge)
* syntax -> Package Syntax Report (badge)
* types -> Package Type Hints Report (badge)
* styles -> Package Code Styles Report (badge)
- notebooks -> Package Notebooks (badge)
- docs -> Package Documentation (badge)
```

### Install package

Create a virtual environment if required and activate it:

```bash
python3 -m virtualenv venv
source venv/bin/activate
```

This package follows the usual `setuptools` flow, installation is as simple as:

```bash
python3 setup.py install
```

This will install dependencies as well (as defined in [`requirements.txt`][200]).

To build a wheel and install from it, then issue:

```bash
nox --session package install
```

### Test package

This package uses [`unittest`][101] to create its test suite,
to run the complete package test suite, issue:

```bash
nox --session tests
```

### Test coverage

This package uses [`coverage`][102] to assess code coverage.
To run the test suite coverage, issue:

```bash
nox --session coverage
```

### Check security

This package uses [`bandit`][101] for security checks.
To check package python security know vulnerabilities, issue:

```bash
nox --session security safety
```

### Check syntax

This package uses [`pylint`][103] and [`flake8`][112] for syntax checks.
To check package python syntax, issue:

```bash
nox --session linter syntax
```

### Check types

This package uses [`mypy`][104] to check types hints.
To check type hints and common errors, issue:

```bash
nox --session types
```

### Check styles

This package uses [`black`][105] and [`isort`][106] to check or coerce python code styles.
To check if your code is black, issue:

```bash
nox --session styles
```

To actually style the package code inplace, issue:

```bash
nox --session clean
```

### Refresh notebooks

This package uses [`jupyter`][107] notebooks for tests and documentation purposes.
To refresh all notebooks, issue:

```bash
nox --session notebooks
```

Note: it will rely on previously defined virtual environment `venv`.


### Build documentation

This package uses [Sphinx][108] to build documentation.
To generate the package documentation, issue:

```bash
nox --session docs
```

### Generate badges

All badges are automatically generated for each [`nox`][110]
session using [`anybadge`][109] and related report
contents (see [`noxfile.py`][210] for details).

[100]: https://github.com/pypa/setuptools
[101]: https://docs.python.org/3/library/unittest.html
[102]: https://github.com/nedbat/coveragepy
[103]: https://github.com/PyCQA/pylint
[104]: https://github.com/python/mypy
[105]: https://github.com/psf/black
[106]: https://github.com/pycqa/isort/
[107]: https://github.com/jupyter/notebook
[108]: https://github.com/sphinx-doc/sphinx
[109]: https://github.com/jongracecox/anybadge
[110]: https://github.com/theacodes/nox
[111]: https://github.com/PyCQA/bandit
[112]: https://github.com/PyCQA/flake8
[113]: https://github.com/initios/flake8-junit-report
[114]: https://github.com/pyupio/safety

[200]: ./requirements.txt
[201]: requirements_ci.txt
[210]: ./noxfile.py

[300]: https://hub.docker.com/r/jlandercy/python-qc/tags?page=1&ordering=last_updated
