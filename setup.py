"""
Package Installer
"""

import pathlib

from setuptools import find_packages, setup

import scifit

_path = pathlib.Path(__file__).resolve().parents[0]
_package = _path.parts[-1]

with (_path / 'requirements.txt').open() as fh:
    requirements = fh.read().splitlines()

setup(
    name=_package,
    version=newproject.__version__,
    url="https://github.com/jlandercy/{package:}".format(package=_package),
    license="BSD 3-Clause License",
    author="Jean Landercy",
    author_email="jeanlandercy@live.com",
    description="Minimal Python 3 Package",

    packages=find_packages(exclude=[]),
    package_data={
       "": ["**/*.json"],
       _package: ["resources/*"]
    },
    scripts=[],
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Topic :: Education",
        "Topic :: Software Development",
        "Topic :: Utilities"
    ],
    entry_points={
        "console_scripts": ["{package:}={package:}._new:main".format(package=_package)]
    },
    zip_safe=False,
)
