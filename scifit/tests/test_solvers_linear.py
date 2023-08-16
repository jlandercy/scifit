from unittest import TestCase

import numpy as np

from scifit.solvers import linear
from scifit.tests.helpers import GenericTestFitSolver


class GenericRegression(GenericTestFitSolver):
    factory = linear.ConstantFitSolver
    parameters = np.array([5.0])


class GenericConstantRegression(GenericTestFitSolver):
    factory = linear.ConstantFitSolver
    parameters = np.array([5.0])
    scale_mode = "abs"


class ConstantRegressionNoiseL0(GenericConstantRegression, TestCase):
    sigma = 1e-6


class ConstantRegressionNoiseL1(GenericConstantRegression, TestCase):
    sigma = 2.5e-2


class ConstantRegressionNoiseL2(GenericConstantRegression, TestCase):
    sigma = 1.0e-1


class ConstantRegressionNoiseL3(GenericConstantRegression, TestCase):
    sigma = 2.5e-1


class ConstantRegressionNoiseL4(GenericConstantRegression, TestCase):
    sigma = 1.0


class ConstantRegressionNoiseL5(GenericConstantRegression, TestCase):
    sigma = 2.5


class GenericProportionalRegression(GenericTestFitSolver):
    factory = linear.ProportionalFitSolver
    parameters = np.array([5.0])


class ProportionalRegressionNoiseL0(GenericProportionalRegression, TestCase):
    sigma = 1e-6


class ProportionalRegressionNoiseL1(GenericProportionalRegression, TestCase):
    sigma = 2.5e-2


class ProportionalRegressionNoiseL2(GenericProportionalRegression, TestCase):
    sigma = 1.0e-1


class ProportionalRegressionNoiseL3(GenericProportionalRegression, TestCase):
    sigma = 2.5e-1


class ProportionalRegressionNoiseL4(GenericProportionalRegression, TestCase):
    sigma = 1.0


class ProportionalRegressionNoiseL5(GenericProportionalRegression, TestCase):
    sigma = 2.5


class GenericLinearRegression(GenericTestFitSolver):
    factory = linear.LinearFitSolver
    parameters = np.array([2.0, 3.0])


class LinearRegressionNoiseL0(GenericLinearRegression, TestCase):
    sigma = 1e-6


class LinearRegressionNoiseL1(GenericLinearRegression, TestCase):
    sigma = 2.5e-2


class LinearRegressionNoiseL2(GenericLinearRegression, TestCase):
    sigma = 1.0e-1


class LinearRegressionNoiseL3(GenericLinearRegression, TestCase):
    sigma = 2.5e-1


class LinearRegressionNoiseL4(GenericLinearRegression, TestCase):
    sigma = 1.0


class LinearRegressionNoiseL5(GenericLinearRegression, TestCase):
    sigma = 2.5


class GenericParabolicRegression(GenericTestFitSolver):
    factory = linear.ParabolicFitSolver
    parameters = np.array([1.0, 2.0, 3.0])


class ParabolicRegressionNoiseL0(GenericParabolicRegression, TestCase):
    sigma = 1e-6


class ParabolicRegressionNoiseL1(GenericParabolicRegression, TestCase):
    sigma = 2.5e-2


class ParabolicRegressionNoiseL2(GenericParabolicRegression, TestCase):
    sigma = 1.0e-1


class ParabolicRegressionNoiseL3(GenericParabolicRegression, TestCase):
    sigma = 2.5e-1


class ParabolicRegressionNoiseL4(GenericParabolicRegression, TestCase):
    sigma = 1.0


class ParabolicRegressionNoiseL5(GenericParabolicRegression, TestCase):
    sigma = 2.5


class GenericCubicRegression(GenericTestFitSolver):
    factory = linear.CubicFitSolver
    parameters = np.array([1.0, 2.0, 3.0, 4.0])


class CubicRegressionNoiseL0(GenericCubicRegression, TestCase):
    sigma = 1e-6


class CubicRegressionNoiseL1(GenericCubicRegression, TestCase):
    sigma = 2.5e-2


class CubicRegressionNoiseL2(GenericCubicRegression, TestCase):
    sigma = 1.0e-1


class CubicRegressionNoiseL3(GenericCubicRegression, TestCase):
    sigma = 2.5e-1


class CubicRegressionNoiseL4(GenericCubicRegression, TestCase):
    sigma = 1.0


class CubicRegressionNoiseL5(GenericCubicRegression, TestCase):
    sigma = 2.5


class GenericLinearRootRegression(GenericTestFitSolver):
    factory = linear.LinearRootFitSolver
    parameters = np.array([1.0, 2.0, 3.0])


class LinearRootRegressionNoiseL0(GenericLinearRootRegression, TestCase):
    sigma = 1e-6


class LinearRootRegressionNoiseL1(GenericLinearRootRegression, TestCase):
    sigma = 2.5e-2


class LinearRootRegressionNoiseL2(GenericLinearRootRegression, TestCase):
    sigma = 1.0e-1


class LinearRootRegressionNoiseL3(GenericLinearRootRegression, TestCase):
    sigma = 2.5e-1


class LinearRootRegressionNoiseL4(GenericLinearRootRegression, TestCase):
    sigma = 1.0


class LinearRootRegressionNoiseL5(GenericLinearRootRegression, TestCase):
    sigma = 2.5


class Generic2DFeatureRegression(GenericTestFitSolver):
    factory = linear.PlaneFitSolver
    parameters = np.array([1.0, 1.0, 1.0])
    dimension = 2
    resolution = 10


class PlaneRegressionNoiseL0(Generic2DFeatureRegression, TestCase):
    sigma = 1e-6


class PlaneRegressionNoiseL1(Generic2DFeatureRegression, TestCase):
    sigma = 2.5e-2


class PlaneRegressionNoiseL2(Generic2DFeatureRegression, TestCase):
    sigma = 1.0e-1


class PlaneRegressionNoiseL3(Generic2DFeatureRegression, TestCase):
    sigma = 2.5e-1


class PlaneRegressionNoiseL4(Generic2DFeatureRegression, TestCase):
    sigma = 1.0


class PlaneRegressionNoiseL5(Generic2DFeatureRegression, TestCase):
    sigma = 2.5


class QuadricRegression(Generic2DFeatureRegression):
    factory = linear.QuadricFitSolver
    parameters = np.array([1.0, -1.0, 1.0])


class SaddleRegressionNoiseL0(QuadricRegression, TestCase):
    sigma = 1e-6


class SaddleRegressionNoiseL1(QuadricRegression, TestCase):
    sigma = 2.5e-2


class SaddleRegressionNoiseL2(QuadricRegression, TestCase):
    sigma = 1.0e-1


class SaddleRegressionNoiseL3(QuadricRegression, TestCase):
    sigma = 2.5e-1


class SaddleRegressionNoiseL4(QuadricRegression, TestCase):
    sigma = 1.0


class SaddleRegressionNoiseL5(QuadricRegression, TestCase):
    sigma = 2.5


class ParaboloidRegressionNoiseL0(QuadricRegression, TestCase):
    parameters = np.array([1.0, 1.0, 1.0])
    sigma = 1e-6


class Paraboloid3RegressionNoiseL0(QuadricRegression, TestCase):
    parameters = np.array([1.0, 0.5, -0.3])
    sigma = 1e-6


class FullQuadricRegression(Generic2DFeatureRegression):
    factory = linear.FullQuadricFitSolver
    parameters = np.array([1.0, -1.0, 3.0, 0.2, 0.2, 1.0])


class FullSaddleRegressionNoiseL0(FullQuadricRegression, TestCase):
    sigma = 1e-6


class FullSaddleRegressionNoiseL1(FullQuadricRegression, TestCase):
    sigma = 2.5e-2


class FullSaddleRegressionNoiseL2(FullQuadricRegression, TestCase):
    sigma = 1.0e-1


class FullSaddleRegressionNoiseL3(FullQuadricRegression, TestCase):
    sigma = 2.5e-1


class FullSaddleRegressionNoiseL4(FullQuadricRegression, TestCase):
    sigma = 1.0


class FullSaddleRegressionNoiseL5(FullQuadricRegression, TestCase):
    sigma = 2.5
    sigma_factor = 25.0
