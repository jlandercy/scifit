"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`scifit.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np

from scifit.solvers import linear
from scifit.tests.helpers import GenericTestFitSolver


class GenericConstantRegression(GenericTestFitSolver):
    factory = linear.ConstantFitSolver
    parameters = np.array([5.])


class ConstantRegressionNoiseL0(GenericConstantRegression, TestCase):
    sigma = None


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
    factory = linear.ProportionalFitSolver
    parameters = np.array([5.])


class ProportionalRegressionNoiseL0(GenericProportionalRegression, TestCase):
    sigma = None


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
    factory = linear.LinearFitSolver
    parameters = np.array([2., 3.])


class LinearRegressionNoiseL0(GenericLinearRegression, TestCase):
    sigma = None


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
    factory = linear.ParabolicFitSolver
    parameters = np.array([1., 2., 3.])


class ParabolicRegressionNoiseL0(GenericParabolicRegression, TestCase):
    sigma = None


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
    factory = linear.CubicFitSolver
    parameters = np.array([1., 2., 3., 4.])


class CubicRegressionNoiseL0(GenericCubicRegression, TestCase):
    sigma = None


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
    factory = linear.LinearRootFitSolver
    parameters = np.array([1., 2., 3.])


class LinearRootRegressionNoiseL0(GenericLinearRootRegression, TestCase):
    sigma = None


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


class Generic2DFeatureRegression(GenericTestFitSolver):
    factory = linear.PlaneFitSolver
    dimension = 2
    resolution = 10
    scale = 200
    parameters = np.array([1., 1., 1.])


class PlaneRegressionNoiseL0(Generic2DFeatureRegression, TestCase):
    sigma = None


class PlaneRegressionNoiseL1(Generic2DFeatureRegression, TestCase):
    sigma = 1.e-3


class PlaneRegressionNoiseL2(Generic2DFeatureRegression, TestCase):
    sigma = 1.e-2


class PlaneRegressionNoiseL3(Generic2DFeatureRegression, TestCase):
    sigma = 1.e-1


class PlaneRegressionNoiseL4(Generic2DFeatureRegression, TestCase):
    sigma = 1.
    scale = 25000


class PlaneRegressionNoiseL5(Generic2DFeatureRegression, TestCase):
    sigma = 10.


class QuadricRegression(Generic2DFeatureRegression):
    factory = linear.QuadricFitSolver
    parameters = np.array([1., -1., 1.])


class SaddleRegressionNoiseL0(QuadricRegression, TestCase):
    sigma = None


class SaddleRegressionNoiseL1(QuadricRegression, TestCase):
    sigma = 1.e-3


class SaddleRegressionNoiseL2(QuadricRegression, TestCase):
    sigma = 1.e-2


class SaddleRegressionNoiseL3(QuadricRegression, TestCase):
    sigma = 1.e-1


class SaddleRegressionNoiseL4(QuadricRegression, TestCase):
    sigma = 1.
    scale = 25000


class SaddleRegressionNoiseL5(QuadricRegression, TestCase):
    sigma = 10.


class ParaboloidRegressionNoiseL0(QuadricRegression, TestCase):
    parameters = np.array([1., 1., 1.])
    sigma = None


class Paraboloid3RegressionNoiseL0(QuadricRegression, TestCase):
    parameters = np.array([1., 0.5, -0.3])
    sigma = None
