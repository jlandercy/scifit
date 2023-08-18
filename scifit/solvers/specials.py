import numpy as np
from scipy import integrate

from scifit.interfaces.generic import FitSolverInterface


class DebyeFitSolver(FitSolverInterface):

    @staticmethod
    def _integral(n):

        def integrand(t):
            return t ** n / (np.exp(t) - 1)

        @np.vectorize
        def wrapped(x):
            return (n / np.power(x, n)) * integrate.quad(integrand, 0, x)[0]

        return wrapped

    @staticmethod
    def model(x, theta):
        return 3 * x[:, 0] * DebyeFitSolver._integral(3)(theta / x[:, 0])

