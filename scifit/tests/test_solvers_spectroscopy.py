import os
import pathlib
from unittest import TestCase

import numpy as np
import pandas as pd

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
        solution = self.solver.fit(self.data)
        axe = self.solver.plot_fit(title=name)
        axe.figure.savefig(
            "{}/{}_fit.{}".format(self.media_path, name, self.format), dpi=120
        )


class TestChromatogramSolver01(TestSpectroscopySolver, TestCase):
    configuration = {}

