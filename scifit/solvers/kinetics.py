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
        solution = cls.kinetic.integrate(x[:, 0], [b0], None).y.T[:, 0]
        return solution


class SequenceKineticSolver(FitSolver1D):
    _model_equation = r"A \overset{\beta_0}{\longrightarrow} B(t) \overset{\beta_1}{\longrightarrow} C"

    kinetic = KineticSolverInterface(
        nur=np.array(
            [
                [-1, 1, 0],
                [0, -1, 1],
            ]
        ),
        k0=np.array([1e-2, 1e-3]),
        x0=np.array([1e-3, 0.0, 0.0]),
    )

    @classmethod
    def model(cls, x, b0, b1):
        """

        Solve sequential intermediate kinetic :math:`B(t)`:

        .. math::

            A \\overset{\\beta_0}{\\longrightarrow} B(t) \\overset{\\beta_1}{\\longrightarrow} C

        And adjust dataset to it:

        .. image:: ../media/figures/SequenceKinetic.png
            :width: 560
            :alt: Sequence Kinetic Adjustment

        :param x:
        :param b0:
        :param b1:
        :return:
        """
        solution = cls.kinetic.integrate_system(x[:, 0], [b0, b1], None).y.T[:, 1]
        return solution


class SequenceOfThreeKineticSolver(FitSolver1D):
    kinetic = KineticSolverInterface(
        nur=np.array(
            [
                [-1, -1, 0, 0],
                [-1, 0, -1, 0],
                [-1, 0, 0, -1],
            ]
        ),
        nup=np.array(
            [
                [2, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
            ]
        ),
        x0=np.array([3e-3, 2e-3, 1e-3, 0.0]),
        k0=np.array([2.1, 1, 9e-1]),
    )

    @classmethod
    def model(cls, x, b0, b1, b2):
        solution = cls.kinetic.integrate_system(x[:, 0], [b0, b1, b2], None).y.T[:, 0]
        return solution


class BrusselatorKineticSolver(FitSolver1D):
    _model_equation = r"""A &\overset{\beta_0}{\longrightarrow}& E(t) \\
    2E(t) + F &\overset{\beta_1}{\longrightarrow}& 3E(t) \\
    B + E(t) &\overset{\beta_2}{\longrightarrow}& F + C \\
    E(t) &\overset{\beta_3}{\longrightarrow}& D"""
    _equation_array = True

    kinetic = KineticSolverInterface(
        nur=np.array(
            [
                [-1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -2, -1],
                [0, -1, 0, 0, -1, 0],
                [0, 0, 0, 0, -1, 0],
            ]
        ),
        nup=np.array(
            [
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 3, 0],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
            ]
        ),
        k0=np.array([1.0, 1.0, 1.0, 1.0]),
        x0=np.array([1.0, 3.0, 0.0, 0.0, 1.0, 1.0]),
        unsteady=np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    )

    @classmethod
    def model(cls, x, b0, b1, b2, b3):
        """
        Solve the Brusselator problem for :math:`E(t)`:

        .. math::

            \\begin{eqnarray}
            A &\\overset{\\beta_0}{\\longrightarrow}& E(t) \\\\
            2E(t) + F &\\overset{\\beta_1}{\\longrightarrow}& 3E(t) \\\\
            B + E(t) &\\overset{\\beta_2}{\\longrightarrow}& F + C \\\\
            E(t) &\\overset{\\beta_3}{\\longrightarrow}& D
            \\end{eqnarray}

        And adjust data to it:

        .. image:: ../media/figures/BrusselatorKinetic.png
            :width: 560
            :alt: Brusselator Kinetic Adjustment

        :param x:
        :param b0:
        :param b1:
        :param b2:
        :param b3:
        :return:
        """
        solution = cls.kinetic.integrate_system(x[:, 0], [b0, b1, b2, b3], None).y.T[
            :, 4
        ]
        return solution
