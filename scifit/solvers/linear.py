import numpy as np

from scifit.interfaces.solvers import FitSolver1D, FitSolver2D


class ConstantFitSolver(FitSolver1D):
    _model_equation = r"y = \beta_0"

    @staticmethod
    def model(x, b0):
        """
        Constant model defined as follows:

        .. math::

            y = \\beta_0
        """
        return np.full(x.shape[0], b0)


class ProportionalFitSolver(FitSolver1D):
    _model_equation = r"y = \beta_0 x"

    @staticmethod
    def model(x, b0):
        """
        Proportional model defined as follows:

        .. math::

            y = \\beta_0 \\cdot x
        """
        return b0 * x[:, 0]


class LinearFitSolver(FitSolver1D):
    _model_equation = r"y = \beta_1 x + \beta_0"

    @staticmethod
    def model(x, b0, b1):
        """
        Linear model defined as follows:

        .. math::

            y = \\beta_1 \\cdot x + \\beta_0
        """
        return b1 * x[:, 0] + b0


class ParabolicFitSolver(FitSolver1D):
    _model_equation = r"y = \beta_2 \cdot x^2 + \beta_1 \cdot x + \beta_0"

    @staticmethod
    def model(x, b0, b1, b2):
        """
        Parabolic model defined as follows:

        .. math::

            y = \\beta_2 \\cdot x^2 + \\beta_1 \\cdot x + \\beta_0
        """
        return b2 * np.power(x[:, 0], 2) + b1 * x[:, 0] + b0


class CubicFitSolver(FitSolver1D):
    _model_equation = (
        r"y = \beta_3 \cdot x^3 + \beta_2 \cdot x^2 + \beta_1 \cdot x + \beta_0"
    )

    @staticmethod
    def model(x, b0, b1, b2, b3):
        """
        Cubic model defined as follows:

        .. math::

            y = \\beta_3 \\cdot x^3 + \\beta_2 \\cdot x^2 + \\beta_1 \\cdot x + \\beta_0
        """
        return b3 * np.power(x[:, 0], 3) + b2 * np.power(x[:, 0], 2) + b1 * x[:, 0] + b0


class LinearRootFitSolver(FitSolver1D):
    _model_equation = r"y = \beta_2 \cdot x + \beta_1 \cdot \sqrt{|x| + 1} + \beta_0"

    @staticmethod
    def model(x, b0, b1, b2):
        """
        Linear Root model defined as follows:

        .. math::

            y = \\beta_2 \\cdot x + \\beta_1 \\cdot \\sqrt{|x| + 1} + \\beta_0
        """
        return b2 * x[:, 0] + b1 * np.sqrt(np.abs(x[:, 0]) + 1.0) + b0


class PlaneFitSolver(FitSolver2D):
    _model_equation = "y = a \cdot x_0 + b \cdot x_1 + c"

    @staticmethod
    def model(x, a, b, c):
        """
        Plane model defined as follows:

        .. math::

            y = a \\cdot x_0 + b \\cdot x_1 + c
        """
        return a * x[:, 0] + b * x[:, 1] + c


class QuadricFitSolver(FitSolver2D):
    _model_equation = r"y = a \cdot x^2_0 + b \cdot x^2_1 + c"

    @staticmethod
    def model(x, a, b, c):
        """
        Quadric model defined as follows (normalized):

        .. math::

            y = a \\cdot x^2_0 + b \\cdot x^2_1 + c
        """
        return a * x[:, 0] * x[:, 0] + b * x[:, 1] * x[:, 1] + c


class FullQuadricFitSolver(FitSolver2D):
    _model_equation = r"y = a \cdot x^2_0 + b \cdot x^2_1 + c \cdot x_0x_1 + d \cdot x_0 + e \cdot x_1 + f"

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
