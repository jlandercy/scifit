import numpy as np

from scifit.interfaces.solvers import FitSolver1D, FitSolver2D


class ConstantFitSolver(FitSolver1D):
    @staticmethod
    def model(x, a):
        """
        Constant model defined as follows:

        .. math::

            y = a
        """
        return np.full(x.shape[0], a)


class ProportionalFitSolver(FitSolver1D):
    @staticmethod
    def model(x, a):
        """
        Proportional model defined as follows:

        .. math::

            y = a \\cdot x
        """
        return a * x[:, 0]


class LinearFitSolver(FitSolver1D):

    _model_equation = r"y = a x + b"

    @staticmethod
    def model(x, a, b):
        """
        Linear model defined as follows:

        .. math::

            y = a \\cdot x + b
        """
        return a * x[:, 0] + b


class ParabolicFitSolver(FitSolver1D):
    @staticmethod
    def model(x, a, b, c):
        """
        Parabolic model defined as follows:

        .. math::

            y = a \\cdot x^2 + b \\cdot x + c
        """
        return a * np.power(x[:, 0], 2) + b * x[:, 0] + c


class CubicFitSolver(FitSolver1D):
    @staticmethod
    def model(x, a, b, c, d):
        """
        Cubic model defined as follows:

        .. math::

            y = a \\cdot x^3 + b \\cdot x^2 + c \\cdot x + d
        """
        return a * np.power(x[:, 0], 3) + b * np.power(x[:, 0], 2) + c * x[:, 0] + d


class LinearRootFitSolver(FitSolver1D):
    @staticmethod
    def model(x, a, b, c):
        """
        Linear Root model defined as follows:

        .. math::

            y = a \\cdot x + b \\cdot \\sqrt{|x| + 1} + c
        """
        return a * x[:, 0] + b * np.sqrt(np.abs(x[:, 0]) + 1.0) + c


class PlaneFitSolver(FitSolver2D):
    @staticmethod
    def model(x, a, b, c):
        """
        Plane model defined as follows:

        .. math::

            y = a \\cdot x_0 + b \\cdot x_1 + c
        """
        return a * x[:, 0] + b * x[:, 1] + c


class QuadricFitSolver(FitSolver2D):
    @staticmethod
    def model(x, a, b, c):
        """
        Quadric model defined as follows (normalized):

        .. math::

            y = a \\cdot x^2_0 + b \\cdot x^2_1 + c
        """
        return a * x[:, 0] * x[:, 0] + b * x[:, 1] * x[:, 1] + c


class FullQuadricFitSolver(FitSolver2D):
    @staticmethod
    def model(x, a, b, c, d, e, f):
        """
        Quadric model defined as follows:

        .. math::

            y = a \\cdot x^2_0 + b \\cdot x^2_1 + c \\cdot x_0x_1 + d \\cdot x_0 + e \\cdot x_1 + f
        """
        return (
            a * x[:, 0] * x[:, 0]
            + b * x[:, 1] * x[:, 1]
            + c * x[:, 0] * x[:, 1]
            + d * x[:, 0]
            + e * x[:, 1]
            + f
        )
