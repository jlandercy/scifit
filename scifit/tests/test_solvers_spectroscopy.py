import os
import pathlib
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from pybaselines import Baseline, utils

from scifit.solvers.spectroscopy import SpectroscopySolver

print_report = bool(int(os.getenv("TESTS_PRINT_REPORT", 0)))


class TestSpectroscopySolver:

    root_path = ".cache/media/tests/"
    format = "png"

    factory = SpectroscopySolver
    seed = 12345

    def setUp(self):
        self.media_path = pathlib.Path(self.root_path) / format(
            self.factory.__module__.split(".")[-1]
        )
        self.media_path.mkdir(parents=True, exist_ok=True)

        self.solver = self.factory()
        self.data = self.solver.synthetic_dataset()

    def test_plot_fit(self):
        name = self.__class__.__name__
        solution = self.solver.fit(self.data, prominence=0.25)
        axe = self.solver.plot_fit(title=name)
        axe.figure.savefig(
            "{}/{}_fit.{}".format(self.media_path, name, self.format), dpi=120
        )


class TestChromatogramSolver01(TestSpectroscopySolver, TestCase):
    configuration = {}


class TestRamanSpectrum(TestCase):

    root_path = ".cache/media/tests/"
    factory = SpectroscopySolver
    format = "png"

    def setUp(self) -> None:
        self.media_path = pathlib.Path(self.root_path) / format(
            self.factory.__module__.split(".")[-1]
        )
        self.media_path.mkdir(parents=True, exist_ok=True)

        self.files = list(pathlib.Path("scifit/tests/features/raman/").glob("*.txt"))

    def test_files(self):

        for file in self.files:

            data = pd.read_csv(file, sep="\t", header=None, skiprows=100, names=["x0", "y"])

            solver = SpectroscopySolver()
            solution = solver.fit(data, prominence=350., distance=1., width=(0.5, 50.0))

            axe = solver.plot_fit()

            axe.figure.savefig(
                "{}/{}_fit.{}".format(self.media_path, file.name, self.format), dpi=120
            )
            plt.close(axe.figure)

