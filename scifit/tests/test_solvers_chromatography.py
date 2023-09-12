import pathlib
from unittest import TestCase

import numpy as np
import pandas as pd

from scifit.solvers.chromatography import ChromatogramSolver


class TestChromatogramSolver:
    root_path = ".cache/media/tests/"
    format = "png"

    factory = ChromatogramSolver
    seed = 12345
    n_peaks = 8
    mode = "exp"
    xmin = 0.0
    xmax = 1000.0
    resolution = 5001
    peaks = None
    heights = None
    widths = None

    def setUp(self):
        self.media_path = pathlib.Path(self.root_path) / format(
            self.factory.__module__.split(".")[-1]
        )
        self.media_path.mkdir(parents=True, exist_ok=True)

        self.solver = self.factory()
        self.data = self.solver.synthetic_dataset(
            peaks=self.peaks,
            heights=self.heights,
            widths=self.widths,
            xmin=self.xmin,
            xmax=self.xmax,
            n_peaks=self.n_peaks,
            resolution=self.resolution,
            seed=self.seed,
            mode=self.mode,
        )

    def test_random_peaks(self):
        peaks = self.solver.random_peaks()
        self.assertIsInstance(peaks, dict)

    def test_synthetic_dataset(self):
        self.assertIsInstance(self.data, pd.DataFrame)

    def test_dump_data(self):
        name = self.__class__.__name__
        self.data.to_csv("scifit/tests/features/peaks/{}.csv".format(name), sep=";")

    def test_fit_data(self):
        solution = self.solver.fit(self.data)
        self.assertIsInstance(solution, dict)
        self.assertEqual(solution["peaks"]["indices"].size, self.n_peaks)

    def test_plot_fit(self):
        name = self.__class__.__name__
        solution = self.solver.fit(self.data)
        axe = self.solver.plot_fit()
        axe.figure.savefig("{}/{}_fit.{}".format(self.media_path, name, self.format))


class TestChromatogramSolverSample00(TestChromatogramSolver, TestCase):
    peaks = [100, 180, 350, 420, 550, 700, 800, 880]
    heights = [9, 6, 8, 15, 6, 13, 9, 9]
    widths = [12, 5, 11, 18, 6, 8, 9, 7]
    n_peaks = len(peaks)


class TestChromatogramSolverSample01(TestChromatogramSolver, TestCase):
    pass


class TestChromatogramSolverSample02(TestChromatogramSolver, TestCase):
    mode = "lin"
