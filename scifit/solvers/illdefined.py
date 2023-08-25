import numpy as np

from scifit.interfaces.solvers import FitSolver1D


class LinearSquaredSlopeSolver(FitSolver1D):
    @staticmethod
    def model(x, a, b):
        return np.power(a, 2) * x[:, 0] + b
