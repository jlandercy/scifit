from unittest import TestCase

import numpy as np
import pandas as pd

from scifit.solvers import kinetics
from scifit.tests.helpers import GenericPlotTestFitSolver, GenericSetupTestFitSolver


class GenericKineticRegression(GenericSetupTestFitSolver, GenericPlotTestFitSolver):
    pass


class SimpleKineticRegression(GenericKineticRegression):
    factory = kinetics.SimpleKineticSolver
    parameters = np.array([2.1e-2])
    configuration = {"p0": np.array([1e-4])}
    xmin = 0.0
    xmax = 500.0
    resolution = 50


class SimpleKineticRegressionNoiseL0(SimpleKineticRegression, TestCase):
    sigma = 1e-6


class SimpleKineticRegressionNoiseL1(SimpleKineticRegression, TestCase):
    sigma = 2.5e-2


class SimpleKineticRegressionNoiseL2(SimpleKineticRegression, TestCase):
    sigma = 1e-1


class SequenceKineticRegression(GenericKineticRegression):
    factory = kinetics.SequenceKineticSolver
    parameters = np.array([2.1e-2, 3.15e-3])
    configuration = {"p0": np.array([1e-3, 1e-3])}
    xmin = 0.0
    xmax = 1000.0
    resolution = 50
    loss_resolution = 20


class SequenceKineticRegressionNoiseL0(SequenceKineticRegression, TestCase):
    sigma = 1e-6


class SequenceKineticRegressionNoiseL1(SequenceKineticRegression, TestCase):
    sigma = 2.5e-2


class SequenceKineticRegressionNoiseL2(SequenceKineticRegression, TestCase):
    sigma = 1e-1


class SequenceOfThreeKineticRegression(GenericKineticRegression):
    factory = kinetics.SequenceOfThreeKineticSolver
    parameters = np.array([2, 1, 1])
    configuration = {"p0": np.array([2, 1, 1])}
    xmin = 0.0
    xmax = 1000.0
    resolution = 75
    loss_resolution = 20


class SequenceOfThreeKineticRegressionNoiseL0(
    SequenceOfThreeKineticRegression, TestCase
):
    sigma = 1e-6


class SequenceOfThreeKineticRegressionNoiseL1(
    SequenceOfThreeKineticRegression, TestCase
):
    sigma = 2.5e-2


class SequenceOfThreeKineticRegressionNoiseL2(
    SequenceOfThreeKineticRegression, TestCase
):
    sigma = 1e-1


class BrusselatorKineticRegression(GenericKineticRegression):
    factory = kinetics.BrusselatorKineticSolver
    parameters = np.array([1.1, 0.9, 1.1, 0.9])
    xmin = 0.0
    xmax = 20.0
    resolution = 75
    loss_resolution = 20

    def test_plot_loss(self):
        """Too heavy"""
        pass

    def test_process_report(self):
        """Too heavy"""
        pass


class BrusselatorKineticRegressionNoiseL0(BrusselatorKineticRegression, TestCase):
    sigma = 1e-6


class BrusselatorKineticRegressionNoiseL1(BrusselatorKineticRegression, TestCase):
    sigma = 2.5e-2


class BrusselatorKineticRegressionNoiseL2(BrusselatorKineticRegression, TestCase):
    sigma = 1e-1
