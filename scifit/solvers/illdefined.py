import numpy as np

from scifit.interfaces.generic import FitSolverInterface


class LinearSquaredSlopeSolver(FitSolverInterface):
    @staticmethod
    def model(x, a, b):
        return np.power(a, 2) * x[:, 0] + b

