import numpy as np

from scifit.interfaces.generic import FitSolverInterface


class GompertzFitSolver(FitSolverInterface):
    """
    `Gompertz function model (calculus) <https://en.wikipedia.org/wiki/Gompertz_function>`_
    """

    @staticmethod
    def model(x, a, b, c):
        """
        Gompertz function is defined as follows:

        .. math::

            y = a \\cdot \\exp\\left( - b \\cdot \\exp \\left( - c \\cdot x\\right) \\right)

        :param x: :math:`x`
        :param a: asymptotic value :math:`a`
        :param b: displacement :math:`b`
        :param c: growth rate :math:`c`
        :return: Gompertz function :math:`y`
        """
        return a * np.exp(-b * np.exp(-c * x[:, 0]))


class LogisticFitSolver(FitSolverInterface):
    """
    `Logistic function model (calculus) <https://en.wikipedia.org/wiki/Logit>`_
    """

    @staticmethod
    def model(x, k, x0):
        """
        Logistic function is defined as follows:

        .. math::

            y = \\frac{1}{1 + \\exp\\left[- k \\cdot (x - x_0)\\right]}

        :param x: independent variable :math:`x`
        :param k: growth rate or sigmoid steepness :math:`k`
        :param x0: displacement or sigmoid inflexion point :math:`x_0`
        :return: Logistic function :math:`y`
        """
        return 1.0 / (1.0 + np.exp(-k * (x[:, 0]) - x0))


class AlgebraicSigmoidFitSolver(FitSolverInterface):
    """
    `Algebric sigmoid function model (calculus) <https://en.wikipedia.org/wiki/Algebraic_function>`_
    """

    @staticmethod
    def model(x, k):
        """
        Logistic function is defined as follows:

        .. math::

            y = \\frac{x}{\\left(1 + |x|^k\\right^{\\frac{1}{k}}}

        :param x: independent variable :math:`x`
        :param k: growth rate or sigmoid steepness :math:`k`
        :return: Logistic function :math:`y`
        """
        return  x[:, 0] / np.power(1.0 + np.power(x[:, 0], k), 1/k)


class MichaelisMentenKineticFitSolver(FitSolverInterface):
    """
    `Michaëlis-Menten kinetic model (biochemistry) <https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics>`_
    """

    @staticmethod
    def model(x, vmax, km):
        """
        Michaëlis-Menten kinetic is defined as follows:

        .. math::

            y = \\frac{v_\\max \\cdot x}{K_\\mathrm{m} + x}

        :param x: Substrate concentration :math:`x`
        :param vmax: Limiting kinetic rate :math:`v_\\max`
        :param km: Michaëlis constant :math:`K_\\mathrm{m}`
        :return: Michaëlis-Menten kinetic rate :math:`y`
        """
        return (vmax * x[:, 0]) / (km + x[:, 0])


class HillEquationFitSolver(FitSolverInterface):
    """
    `Hill equation model (biochemistry) <https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)>`_
    """

    @staticmethod
    def model(x, n, k):
        """
        Hill equation is defined as follows:

        .. math::

            y = \\frac{k \\cdot x^n}{1 + k \\cdot x^n}

        :param x: total ligand concentration :math:`x`
        :param n: Hill coefficient :math:`n`
        :param k: apparent dissociation constant :math:`k` (law of mass action)
        :return: fraction :math:`y` of the receptor protein concentration that is bound by the ligand
        """
        term = k * np.power(x[:, 0], n)
        return term / (1 + term)
