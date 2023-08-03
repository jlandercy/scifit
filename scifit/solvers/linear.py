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


class ParabolicFitSolver(FitSolverInterface):

    @staticmethod
    def model(x, a, b, c):
        return a * np.power(x[:, 0], 2) + b * x[:, 0] + c


class CubicFitSolver(FitSolverInterface):

    @staticmethod
    def model(x, a, b, c, d):
        return a * np.power(x[:, 0], 3) + b * np.power(x[:, 0], 2) + c * x[:, 0] + d


class LinearRootFitSolver(FitSolverInterface):

    @staticmethod
    def model(x, a, b, c):
        return a * x[:, 0] + b * np.sqrt(np.abs(x[:, 0]) + 1.) + c

