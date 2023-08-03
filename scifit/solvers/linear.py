import numpy as np

from scifit.interfaces.generic import FitSolverInterface


class ConstantFitSolver(FitSolverInterface):

    @staticmethod
    def model(x, a):
        return np.full(x.shape[0], a)


class ProportionalFitSolver(FitSolverInterface):

    @staticmethod
    def model(x, a):
        return a * x[:, 0]


class LinearFitSolver(FitSolverInterface):

    @staticmethod
    def model(x, a, b):
        return a * x[:, 0] + b




