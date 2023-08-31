import numpy as np

from scifit.interfaces.kinetics import KineticSolverInterface
from scifit.interfaces.solvers import FitSolver1D


class SimpleKineticSolver(FitSolver1D):

    _model_equation = r"A(t) \overset{\beta_0}{\longrightarrow} B"

    kinetic = KineticSolverInterface(
        nur=np.array([[-1, 1]]),
        k0=np.array([1e-2]),
        x0=np.array([1e-3, 0.0]),
    )

    @classmethod
    def model(cls, x, b0):
        solution = cls.kinetic.solve(x[:, 0], [b0], None).y.T[:, 0]
        return solution


class SequenceKineticSolver(FitSolver1D):

    _model_equation = r"A \overset{\beta_0}{\longrightarrow} B(t) \overset{\beta_1}{\longrightarrow} C"

    kinetic = KineticSolverInterface(
        nur=np.array([
            [-1, 1, 0],
            [0, -1, 1],
        ]),
        k0=np.array([1e-2, 1e-3]),
        x0=np.array([1e-3, 0.0, 0.0]),
    )

    @classmethod
    def model(cls, x, b0, b1):
        solution = cls.kinetic.solve(x[:, 0], [b0, b1], None).y.T[:, 1]
        return solution
