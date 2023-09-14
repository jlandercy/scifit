import os
import pathlib
from unittest import TestCase

import numpy as np
import pandas as pd

from scifit.solvers.chromatography import ChromatogramSolver


print_report = bool(int(os.getenv("TESTS_PRINT_REPORT", 0)))


class TestChromatogramSolver:
    root_path = ".cache/media/tests/"
    format = "png"

    factory = ChromatogramSolver
    seed = 12345
    n_peaks = 8
    baseline_mode = "exp"
    xmin = 0.0
    xmax = 1000.0
    resolution = 5001
    peaks = None
    heights = None
    widths = None
    b0 = 5.0
    b1 = 15.0

    filter_mode = "imodpoly"
    configuration = {}
    prominence = 2.5
    width = 10.0
    height = None
    distance = None

    def setUp(self):
        self.media_path = pathlib.Path(self.root_path) / format(
            self.factory.__module__.split(".")[-1]
        )
        self.media_path.mkdir(parents=True, exist_ok=True)

        self.solver = self.factory(
            mode=self.filter_mode,
            prominence=self.prominence,
            width=self.width,
            height=self.height,
            distance=self.distance,
            configuration=self.configuration,
        )
        self.data = self.solver.synthetic_dataset(
            peaks=self.peaks,
            heights=self.heights,
            widths=self.widths,
            xmin=self.xmin,
            xmax=self.xmax,
            n_peaks=self.n_peaks,
            resolution=self.resolution,
            seed=self.seed,
            mode=self.baseline_mode,
            b0=self.b0,
            b1=self.b1,
        )

    def test_random_peaks(self):
        peaks = self.solver.random_peaks()
        self.assertIsInstance(peaks, dict)

    def test_synthetic_dataset(self):
        self.assertIsInstance(self.data, pd.DataFrame)

    def test_dump_data(self):
        name = self.__class__.__name__
        self.data.to_csv("{}/{}_data.csv".format(self.media_path, name), sep=";")

    def test_summary(self):
        name = self.__class__.__name__
        self.solver.fit(self.data)
        summary = self.solver.summary()
        summary.to_csv("{}/{}_summary.csv".format(self.media_path, name), sep=";")

    def test_fit_data(self):
        solution = self.solver.fit(self.data)
        self.assertIsInstance(solution, dict)
        self.assertEqual(solution["peaks"]["indices"].size, self.n_peaks)

    def test_plot_fit(self):
        name = self.__class__.__name__
        solution = self.solver.fit(self.data)
        axe = self.solver.plot_fit(title=name)
        axe.figure.savefig(
            "{}/{}_fit.{}".format(self.media_path, name, self.format), dpi=120
        )

    def test_process_report(self):
        if print_report:
            name = self.__class__.__name__
            file = r"{}_report".format(name)
            self.solver.fit(self.data)
            self.solver.report(file=file, path=self.media_path, mode="pdf")


# class TestChromatogramSolverSample00Default(TestChromatogramSolver, TestCase):
#     peaks = [100, 180, 350, 420, 550, 700, 800, 880]
#     heights = [9, 6, 8, 15, 6, 13, 9, 9]
#     widths = [12, 5, 11, 18, 6, 8, 9, 7]
#     n_peaks = len(peaks)


class TestChromatogramSolverSample01Default(TestChromatogramSolver, TestCase):
    configuration = {"poly_order": 3}


# class TestChromatogramSolverSample02Default(TestChromatogramSolver, TestCase):
#     baseline_mode = "lin"
#     configuration = {"poly_order": 3}
#
#
# class TestChromatogramSolverSample03Default(TestChromatogramSolver, TestCase):
#     baseline_mode = "lin"
#     configuration = {"poly_order": 1}
#     b0 = 0.0
#     b1 = 0.0


class TestChromatogramSolverSample01Loess2(TestChromatogramSolver, TestCase):
    filter_mode = "loess"
    configuration = {"poly_order": 2}


# class TestChromatogramSolverSample01ModPoly3(TestChromatogramSolver, TestCase):
#     filter_mode = "modpoly"
#     configuration = {"poly_order": 3}
#
#
# class TestChromatogramSolverSample01GoldIndec3(TestChromatogramSolver, TestCase):
#     filter_mode = "goldindec"
#     configuration = {"poly_order": 3}
#
#
# class TestChromatogramSolverSample01PenalizedPoly1(TestChromatogramSolver, TestCase):
#     filter_mode = "penalized_poly"
#     configuration = {"poly_order": 1}
#
#
# class TestChromatogramSolverSample01PenalizedPoly2(TestChromatogramSolver, TestCase):
#     filter_mode = "penalized_poly"
#     configuration = {"poly_order": 2}


class TestChromatogramSolverSample01PenalizedPoly3(TestChromatogramSolver, TestCase):
    filter_mode = "penalized_poly"
    configuration = {"poly_order": 3}


#
# class TestChromatogramSolverSample01QuantReg3(TestChromatogramSolver, TestCase):
#     filter_mode = "quant_reg"
#     configuration = {"poly_order": 3}
#
#
# class TestChromatogramSolverSample01BEADS(TestChromatogramSolver, TestCase):
#     filter_mode = "beads"
#     configuration = {}
#
#
# class TestChromatogramSolverSample01CWTBR(TestChromatogramSolver, TestCase):
#     filter_mode = "cwt_br"
#
#
# class TestChromatogramSolverSample01Dietrich(TestChromatogramSolver, TestCase):
#     filter_mode = "dietrich"
#
#
# class TestChromatogramSolverSample01FABC(TestChromatogramSolver, TestCase):
#     filter_mode = "fabc"
#


class TestChromatogramSolverSample01FastChrom(TestChromatogramSolver, TestCase):
    filter_mode = "fastchrom"


#
#
# class TestChromatogramSolverSample01Golotvin(TestChromatogramSolver, TestCase):
#     filter_mode = "golotvin"
#
#
# class TestChromatogramSolverSample01StdDistribution(TestChromatogramSolver, TestCase):
#     filter_mode = "std_distribution"
#
#
# class TestChromatogramSolverSample01Amormol(TestChromatogramSolver, TestCase):
#     filter_mode = "amormol"
#
#
# class TestChromatogramSolverSample01IMOR(TestChromatogramSolver, TestCase):
#     filter_mode = "imor"
#
#
# class TestChromatogramSolverSample01JBCD(TestChromatogramSolver, TestCase):
#     filter_mode = "jbcd"
#
#
# class TestChromatogramSolverSample01MOR(TestChromatogramSolver, TestCase):
#     filter_mode = "mor"
#
#
# class TestChromatogramSolverSample01MORMOL(TestChromatogramSolver, TestCase):
#     filter_mode = "mormol"
#
#
# class TestChromatogramSolverSample01MPLS(TestChromatogramSolver, TestCase):
#     filter_mode = "mpls"
#
#
# class TestChromatogramSolverSample01MPSpline(TestChromatogramSolver, TestCase):
#     filter_mode = "mpspline"
#
#
# class TestChromatogramSolverSample01MWMV(TestChromatogramSolver, TestCase):
#     filter_mode = "mwmv"
#
#
# class TestChromatogramSolverSample01RollingBall(TestChromatogramSolver, TestCase):
#     filter_mode = "rolling_ball"
#
#
# class TestChromatogramSolverSample01TopHat(TestChromatogramSolver, TestCase):
#     filter_mode = "tophat"
#
#
# class TestChromatogramSolverSample01Ipsa(TestChromatogramSolver, TestCase):
#     filter_mode = "ipsa"
#
#
# class TestChromatogramSolverSample01NoiseMedian(TestChromatogramSolver, TestCase):
#     filter_mode = "noise_median"
#
#
# class TestChromatogramSolverSample01RIA(TestChromatogramSolver, TestCase):
#     filter_mode = "ria"
#


class TestChromatogramSolverSample01SNIP(TestChromatogramSolver, TestCase):
    filter_mode = "snip"


#
# class TestChromatogramSolverSample01SWiMA(TestChromatogramSolver, TestCase):
#     filter_mode = "swima"
