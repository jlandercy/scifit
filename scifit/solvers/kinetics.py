import numpy as np

from scifit.interfaces.kinetics import KineticSolverInterface
from scifit.interfaces.solvers import FitSolver1D


class SimpleKineticSolver(FitSolver1D):

    kinetic = KineticSolverInterface(
        nur=np.array([[-1, 1]]),
        k0=np.array([1e-2]),
        x0=np.array([1e-3, 0.0]),
    )

    @classmethod
    def model(cls, x, k0):
        solution = cls.kinetic.solve(x[:, 0], [k0], None).y.T[:, 0]
        return solution


class SequenceKineticSolver(FitSolver1D):

    kinetic = KineticSolverInterface(
        nur=np.array([
            [-1, 1, 0],
            [0, -1, 1],
        ]),
        k0=np.array([1e-2, 1e-3]),
        x0=np.array([1e-3, 0.0, 0.0]),
    )

    @classmethod
    def model(cls, x, k0, k1):
        solution = cls.kinetic.solve(x[:, 0], [k0, k1], None).y.T[:, 1]
        return solution
