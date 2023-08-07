import numpy as np

from scifit.interfaces.generic import FitSolverInterface


class GompertzFitSolver(FitSolverInterface):

    @staticmethod
    def model(x, a, b, c):
        return a * np.exp(-b * np.exp(-c * x[:, 0]))


class LogGompertzFitSolver(FitSolverInterface):

    @staticmethod
    def model(x, a, b, c):
        return np.log(a) - b * np.exp(-c * x[:, 0])


class MichaelisMentenKineticFitSolver(FitSolverInterface):

    @staticmethod
    def model(x, vmax, km):
        return (vmax * x[:, 0])/(km + x[:, 0])


class HillKineticFitSolver(FitSolverInterface):

    @staticmethod
    def model(x, n, k):
        term = k * np.power(x[:, 0], n)
        return term/(1 + term)