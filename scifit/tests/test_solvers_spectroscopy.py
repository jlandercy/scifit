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

            data = pd.read_csv(file, sep="\t", header=None, names=["x0", "y"])
            #data = data.set_index("x").apply(lambda x: (x.max() - x) / (x.max() - x.min())).reset_index()
            data = data.iloc[100:, :]

            solver = SpectroscopySolver()
            solution = solver.fit(data, prominence=400.)

            fig, axe = plt.subplots()
            axe.plot(data.x0, data.y, label="Data")
            axe.scatter(data.x0.values[solution["indices"]], data.y.values[solution["indices"]], color="red", label="Peak")
            axe.plot(data.x0, solution["yhat"] + solution["baseline"], label="Fit")
            axe.plot(data.x0, solution["baseline"], "--", label="Baseline")
            axe.legend()
            axe.grid()

            fig.savefig(
                "{}/{}_fit.{}".format(self.media_path, file.name, self.format), dpi=120
            )
            plt.close(fig)

