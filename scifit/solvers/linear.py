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


class ParabolaFitSolver(FitSolverInterface):

    @staticmethod
    def model(x, a, b, c):
        return a * np.power(x[:, 0], 2) + b * x[:, 0] + c


