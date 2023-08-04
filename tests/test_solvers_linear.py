"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`scifit.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

from scifit.solvers.linear import *


class GenericTestFitSolver:

    seed = 789
    factory = LinearFitSolver
    configuration = {}
    parameters = np.array([2., 3.])
    xdata = np.linspace(-1., 1., 30).reshape(-1, 1)
    sigma = 0.

    def setUp(self) -> None:
        self.solver = self.factory(**self.configuration)
        self.ydata = self.solver.target_dataset(
            self.xdata, *self.parameters, sigma=self.sigma,
            proportional=True, generator=np.random.uniform, low=-.5, high=.5,
            seed=self.seed
        )

    def test_signature(self):
        s = self.solver.signature
        n = self.solver.parameter_space_size
        self.assertEqual(len(s.parameters) - 1, n)

    def test_model_implementation(self):
        """
        Check noisy data is close enough to regressed data up to 10 StdDev of applied noise
        Very unlikely to fail but tight enough to detect bad regression
        """
        yhat = self.solver.model(self.xdata, *self.parameters)
        self.assertTrue(np.allclose(self.ydata, yhat, rtol=10 * self.sigma * np.abs(self.ydata)))

    def test_model_fit_signature(self):
        solution = self.solver.fit(self.xdata, self.ydata)
        self.assertIsInstance(solution, dict)
        self.assertSetEqual({"parameters", "covariance", "info", "message", "status"}, set(solution.keys()))
        for key in ["_xdata", "_ydata", "_solution", "_yhat", "_score"]:
            self.assertTrue(hasattr(self.solver, key))

    def test_model_fit_parameters(self):
        """
        Check regressed parameters are equals up to 10 StdDev of fit precision
        Very unlikely to fail but tight enough to detect bad regression
        """
        solution = self.solver.fit(self.xdata, self.ydata)
        for i in range(self.parameters.shape[0]):
            self.assertTrue(
                np.allclose(
                    self.parameters[i],
                    solution["parameters"][i],
                    atol=10*np.sqrt(solution["covariance"][i][i])
                )
            )

    def test_feature_dataset_auto(self):
        self.solver.store(self.xdata, self.ydata)
        dataset = self.solver.feature_dataset(domains=self.solver.feature_domains(), resolution=10)
        self.assertIsInstance(dataset, np.ndarray)
        self.assertEqual(dataset.ndim, 2)
        self.assertEqual(dataset.shape[0], 10**self.solver.m)
        self.assertEqual(dataset.shape[1], self.solver.m)

    def test_parameters_domain_linear_auto(self):
        solution = self.solver.fit(self.xdata, self.ydata)
        domains = self.solver.parameter_domains()

    def test_parameters_domain_linear_fixed(self):
        solution = self.solver.fit(self.xdata, self.ydata)
        domains = self.solver.parameter_domains(xmax=100.)

    def test_parameters_domain_logarithmic_auto(self):
        solution = self.solver.fit(self.xdata, self.ydata)
        domains = self.solver.parameter_domains(mode="log")

    def test_parameters_domain_logarithmic_fixed(self):
        solution = self.solver.fit(self.xdata, self.ydata)
        domains = self.solver.parameter_domains(mode="log", xmax=100.)

    def test_plot_fit(self):
        name = self.__class__.__name__
        title = "{} (seed={:d})".format(name, self.seed)
        self.solver.fit(self.xdata, self.ydata)
        for i, axe in enumerate(self.solver.plot_fit(title=title)):
            axe.figure.savefig("media/{}_fit_x{}.png".format(name, i))
            plt.close(axe.figure)

    def test_plot_loss(self):
        name = self.__class__.__name__
        title = "{} (seed={:d})".format(name, self.seed)
        self.solver.fit(self.xdata, self.ydata)
        for i, axe in enumerate(self.solver.plot_loss(title=title)):
            axe.figure.savefig("media/{}_score_b{}_b{}.png".format(name, *axe._pair_indices))
            plt.close(axe.figure)


class GenericConstantRegression(GenericTestFitSolver):
    factory = ConstantFitSolver
    parameters = np.array([5.])


class ConstantRegressionNoiseL0(GenericConstantRegression, TestCase):
    sigma = 0.


class ConstantRegressionNoiseL1(GenericConstantRegression, TestCase):
    sigma = 1.e-3


class ConstantRegressionNoiseL2(GenericConstantRegression, TestCase):
    sigma = 1.e-2


class ConstantRegressionNoiseL3(GenericConstantRegression, TestCase):
    sigma = 1.e-1


class ConstantRegressionNoiseL4(GenericConstantRegression, TestCase):
    sigma = 1.


class ConstantRegressionNoiseL5(GenericConstantRegression, TestCase):
    sigma = 10.


class GenericProportionalRegression(GenericTestFitSolver):
    factory = ProportionalFitSolver
    parameters = np.array([5.])


class ProportionalRegressionNoiseL0(GenericProportionalRegression, TestCase):
    sigma = 0.


class ProportionalRegressionNoiseL1(GenericProportionalRegression, TestCase):
    sigma = 1.e-3


class ProportionalRegressionNoiseL2(GenericProportionalRegression, TestCase):
    sigma = 1.e-2


class ProportionalRegressionNoiseL3(GenericProportionalRegression, TestCase):
    sigma = 1.e-1


class ProportionalRegressionNoiseL4(GenericProportionalRegression, TestCase):
    sigma = 1.


class ProportionalRegressionNoiseL5(GenericProportionalRegression, TestCase):
    sigma = 10.


class GenericLinearRegression(GenericTestFitSolver):
    factory = LinearFitSolver
    parameters = np.array([2., 3.])


class LinearRegressionNoiseL0(GenericLinearRegression, TestCase):
    sigma = 0.


class LinearRegressionNoiseL1(GenericLinearRegression, TestCase):
    sigma = 1.e-3


class LinearRegressionNoiseL2(GenericLinearRegression, TestCase):
    sigma = 1.e-2


class LinearRegressionNoiseL3(GenericLinearRegression, TestCase):
    sigma = 1.e-1


class LinearRegressionNoiseL4(GenericLinearRegression, TestCase):
    sigma = 1.


class LinearRegressionNoiseL5(GenericLinearRegression, TestCase):
    sigma = 10.


class GenericParabolicRegression(GenericTestFitSolver):
    factory = ParabolicFitSolver
    parameters = np.array([1., 2., 3.])


class ParabolicRegressionNoiseL0(GenericParabolicRegression, TestCase):
    sigma = 0.


class ParabolicRegressionNoiseL1(GenericParabolicRegression, TestCase):
    sigma = 1.e-3


class ParabolicRegressionNoiseL2(GenericParabolicRegression, TestCase):
    sigma = 1.e-2


class ParabolicRegressionNoiseL3(GenericParabolicRegression, TestCase):
    sigma = 1.e-1


class ParabolicRegressionNoiseL4(GenericParabolicRegression, TestCase):
    sigma = 1.


class ParabolicRegressionNoiseL5(GenericParabolicRegression, TestCase):
    sigma = 10.


class GenericCubicRegression(GenericTestFitSolver):
    factory = CubicFitSolver
    parameters = np.array([1., 2., 3., 4.])


class CubicRegressionNoiseL0(GenericCubicRegression, TestCase):
    sigma = 0.


class CubicRegressionNoiseL1(GenericCubicRegression, TestCase):
    sigma = 1.e-3


class CubicRegressionNoiseL2(GenericCubicRegression, TestCase):
    sigma = 1.e-2


class CubicRegressionNoiseL3(GenericCubicRegression, TestCase):
    sigma = 1.e-1


class CubicRegressionNoiseL4(GenericCubicRegression, TestCase):
    sigma = 1.


class CubicRegressionNoiseL5(GenericCubicRegression, TestCase):
    sigma = 10.


class GenericLinearRootRegression(GenericTestFitSolver):
    factory = LinearRootFitSolver
    parameters = np.array([1., 2., 3.])


class LinearRootRegressionNoiseL0(GenericLinearRootRegression, TestCase):
    sigma = 0.


class LinearRootRegressionNoiseL1(GenericLinearRootRegression, TestCase):
    sigma = 1.e-3


class LinearRootRegressionNoiseL2(GenericLinearRootRegression, TestCase):
    sigma = 1.e-2


class LinearRootRegressionNoiseL3(GenericLinearRootRegression, TestCase):
    sigma = 1.e-1


class LinearRootRegressionNoiseL4(GenericLinearRootRegression, TestCase):
    sigma = 1.


class LinearRootRegressionNoiseL5(GenericLinearRootRegression, TestCase):
    sigma = 10.

