import numpy as np

from scifit.interfaces.solvers import FitSolver1D


class LinearSquaredSlopeSolver(FitSolver1D):
    @staticmethod
    def model(x, a, b):
        return np.power(a, 2) * x[:, 0] + b


class SumDiffSquareSolver(FitSolver1D):
    @staticmethod
    def model(x, a, b):
        return (a + b) * np.power(x[:, 0], 2) + (a - b) * x[:, 0] - 1.


class StrangeSOExpCosSolver(FitSolver1D):
    @staticmethod
    def model(x, i0, a, w1, ph):
        return (
            i0
            + (1 - i0) * np.exp(-x[:, 0] / w1)
            + a * np.cos((2 * np.pi / w1) * x[:, 0] - ph)
        )


class SOExpCosSolver(FitSolver1D):
    @staticmethod
    def model(x, a, b, c, d, e, f):
        return (
            a * x[:, 0]
            + b
            + c * np.exp(d * x[:, 0]) * np.cos(2 * np.pi * e * x[:, 0] + f)
        )
