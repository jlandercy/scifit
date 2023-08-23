import numpy as np
from scipy import stats

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
        return 1.0 / (1.0 + np.exp(-k * (x[:, 0] - x0)))


class AlgebraicSigmoidFitSolver(FitSolverInterface):
    """
    `Algebraic sigmoid function model (calculus) <https://en.wikipedia.org/wiki/Algebraic_function>`_
    """

    @staticmethod
    def model(x, k):
        """
        Logistic function is defined as follows:

        .. math::

            y = \\frac{x}{\\left(1 + |x|^k\\right)^{\\frac{1}{k}}}

        :param x: independent variable :math:`x`
        :param k: growth rate or sigmoid steepness :math:`k`
        :return: Algebraic sigmoid function :math:`y`
        """
        return x[:, 0] / np.power(1.0 + np.power(np.abs(x[:, 0]), k), 1.0 / k)


class RichardGeneralizedSigmoidFitSolver(FitSolverInterface):
    """
    `Richard's generalized sigmoid function model (calculus) <https://en.wikipedia.org/wiki/Generalised_logistic_function>`_
    """

    @staticmethod
    def model(x, A, B, C, K, Q, nu):
        """
        Richard's generalized sigmoid (GRS) function is defined as follows:

        .. math::

            y = A + \\frac{K - A}{\\left(C + Q\\cdot\\exp(-B\\cdot t)\\right)^{\\frac{1}{\\nu}}}

        :param A: lower (left) asymptote :math:`A`
        :param B: growth rate or sigmoid steepness :math:`B`
        :param C: asymptotical parameter :math:`C`
        :param K: upper (right) asymptote :math:`K`
        :param Q: location parameter :math:`Q`
        :param nu: asymptotical growth rate :math:`\\nu`
        :return: GRS function :math:`y`
        """
        return A + (K - A) / np.power(C + Q * np.exp(-B * x[:, 0]), 1.0 / nu)


class SmoothstepSigmoidFitSolver(FitSolverInterface):
    """
    `Smoothstep sigmoid function model (calculus) <https://en.wikipedia.org/wiki/Smoothstep>`_
    """

    @staticmethod
    def model(x, a, b):
        """
        Smoothstep function is defined as follows:

        .. math::

            y =
            \\begin{cases}
            0,           & x \\le 0 \\\\
            a \\cdot x^2 - b \\cdot x^3, & 0 \\le x \\le 1 \\\\
            1,           & x \\ge 1 \\\\
            \\end{cases}

        :param a: quadratic coefficient :math:`a`
        :param b: cubic coefficient :math:`b`
        :return: Smoothstep sigmoid function :math:`y`
        """
        y = a * np.power(x[:, 0], 2) - b * np.power(x[:, 0], 3)
        y[x[:, 0] <= 0.0] = 0.0
        y[x[:, 0] >= 1.0] = 1.0
        return y


class InverseBoxCoxFitSolver(FitSolverInterface):
    """
    `Inverse Box-Cox model (calculus) <https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation>`_
    """

    @staticmethod
    def model(x, lambda_):
        """
        Inverse Box-Cox function is defined as follows:

        .. math::

            \\varphi(x, \\lambda) = y =
            \\begin{cases}
            (1 - \\lambda x)^\\frac{1}{\\lambda},  & \\lambda \\ne 0 \\\\
            \\exp(-x),           & \\lambda \\ge 0 \\\\
            \\end{cases}

        :param lambda\_: Box-Cox parameter :math:`\\lambda`
        :return: Inverse Box-Cox function :math:`y`
        """
        if np.allclose(lambda_, 0.0):
            return np.exp(-x[:, 0])
        else:
            return np.power(1.0 - lambda_ * x[:, 0], 1.0 / lambda_)


class DoubleInverseBoxCoxSigmoidFitSolver(FitSolverInterface):
    """
    Double Inverse Box-Cox sigmoid model (calculus)
    """

    @staticmethod
    def model(x, alpha, beta):
        """
        Inverse Box-Cox function is defined as follows:

        .. math::

            y = \\varphi(\\varphi(x, \\beta), \\alpha), \\quad \\alpha < 1, \\, \\beta < 1

        Where :math:`\\varphi` is the Inverse Box-Cox transformation, see :class:`InverseBoxCoxFitSolver` for details.

        :param alpha: first Box-Cox parameter :math:`\\alpha`
        :param beta: second Box-Cox parameter :math:`\\beta`
        :return: Double Inverse Box-Cox sigmoid function :math:`y`
        """
        return InverseBoxCoxFitSolver.model(
            InverseBoxCoxFitSolver.model(x, beta).reshape(-1, 1), alpha
        )


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


class GaussianPeakFitSolver(FitSolverInterface):
    """
    `Gaussian function model (calculus) <https://en.wikipedia.org/wiki/Gaussian_function>`_
    """

    @staticmethod
    def model(x, H, sigma, x0):
        """
        Gaussian function is defined as follows:

        .. math::

            y = H \\exp\\left[- \\frac{1}{2} \\cdot \\left(\\frac{x - x_0}{\\sigma}\\right)^2\\right]

        :param x: independent variable :math:`x`
        :param H: Peak height
        :param sigma: Peak width factor
        :param x0: Peak displacement :math:`x_0`
        :return: Gaussian peak :math:`y`
        """
        return H*np.exp(-0.5 * np.power((x[:, 0] - x0)/sigma, 2))


class GaussianPeakWithBaselineFitSolver(FitSolverInterface):
    """
    Gaussian peak with baseline
    """

    @staticmethod
    def model(x, H, sigma, x0, a, b):
        """
        Gaussian function is defined as follows:

        .. math::

            y = H \\exp\\left[- \\frac{1}{2} \\cdot \\left(\\frac{x - x_0}{\\sigma}\\right)^2\\right] + a x + b

        :param x: independent variable :math:`x`
        :param H: Peak height
        :param sigma: Peak width factor
        :param x0: Peak displacement :math:`x_0`
        :param a:
        :param b:
        :return: Gaussian peak :math:`y`
        """
        return H*np.exp(-0.5 * np.power((x[:, 0] - x0)/sigma, 2)) + a * x[:, 0] + b


class EMGPeakFitSolver(FitSolverInterface):
    """
    `Exponential Modified Gaussian Peak function model (calculus) <https://www.researchgate.net/publication/231172511_Equations_for_chromatographic_peak_modeling_and_calculation_of_peak_area>`_
    """

    cdf = np.vectorize(stats.norm(loc=0.0, scale=1.0).cdf)

    @staticmethod
    def model(x, A, sigma, x0, xG):
        """
        EMG Peak function is defined as follows:

        .. math::

            y = \\frac{A}{x_0} \\exp\\left[-\\frac{1}{2}\\left(\\frac{\\sigma_G}{x_0}\\right)^2 -\\frac{x - x_G}{x_0} \\right] \\cdot \\frac{1}{\\sqrt{2\pi}}\\int\\limits_{\\infty}^{z} \\exp\\left[-\\frac{y^2}{2}\\right]\\mathrm{d}y

        .. math::

            z = \\frac{x - x_G}{\\sigma_G} - \\frac{\\sigma_G}{x_0}

        :param x: independent variable :math:`x`
        :param A:
        :param sigma:
        :param x0:
        :param xG:
        :return: EMG peak :math:`y`
        """

        z = (x[:, 0] - xG)/sigma - sigma/x0
        return A/x0 * np.exp(-0.5 * np.power(sigma/x0, 2) - (x[:, 0] - xG)/x0) * EMGPeakFitSolver.cdf(z)


