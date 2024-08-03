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


class SODoubleExpSolver(FitSolver1D):

    @staticmethod
    def func1(x, n, xe, ye):
        b = 1.9992 * n - 0.3271
        return (ye * 1e-10) * np.exp(-b * np.power(x / xe, 1. / n) - 1.)

    @staticmethod
    def func2(x, y0, h):
        return (y0 * 1e-12) * np.exp(-x / h)

    @staticmethod
    def model(x, n, xe, ye, y0, h):
        return SODoubleExpSolver.func1(x[:, 0], n, xe, ye) + SODoubleExpSolver.func2(x[:, 0], y0, h)
