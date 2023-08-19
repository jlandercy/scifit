import numpy as np
from scipy import integrate, optimize

from scifit.interfaces.generic import FitSolverInterface


class DebyeInternalEnergyFitSolver(FitSolverInterface):
    @staticmethod
    def debye_integral(n):
        """
        Wrapper: Parametrize and integrate the Debye Integral of order n

        .. math::

            D_n(x) = \\frac{n}{x^n} \\int\\limits_0^x \\frac{t^n}{e^t - 1}\\,\\mathrm{d}t

        :param n:
        :return:
        """

        def integrand(t):
            return t**n / (np.exp(t) - 1)

        @np.vectorize
        def wrapped(x):
            return (n / np.power(x, n)) * integrate.quad(integrand, 0, x)[0]

        return wrapped

    @staticmethod
    def model(x, T_D):
        """
        Debye Heat Capacity function is defined as follows:

        .. math::

            \\frac{U}{Nk} = 9T \\left({T\\over T_{\\rm D}}\\right)^3\\int\\limits_0^{T_{\\rm D}/T} {x^3\\over e^x-1}\\, \\mathrm{d}x = 3T D_3 \\left({T_{\\rm D}\\over T}\\right)

        :param x: Temperature
        :param T_D: Debye Temperature
        :return: Molecular Internal Energy
        """
        return (
            3 * x[:, 0] * DebyeInternalEnergyFitSolver.debye_integral(3)(T_D / x[:, 0])
        )


class CrankEquationSolver:
    def __init__(self, alpha):
        self.alpha = alpha
        self.roots = {}

    def lhs(self, x):
        return np.tan(x)

    def rhs(self, x):
        return 3 * x / (3 + self.alpha * x**2)

    def interval(self, n):
        return -np.pi / 2 + n * np.pi, np.pi / 2 + n * np.pi

    def objective(self, x):
        return (self.lhs(x) - self.rhs(x)) ** 2

    def root(self, n):
        if n not in self.roots:
            result = optimize.minimize_scalar(
                self.objective, method="bounded", bounds=self.interval(n)
            )
            self.roots[n] = result.x
        return self.roots[n]

    def compute(self, n):
        solutions = []
        for i in range(n):
            solutions.append(self.root(i + 1))
        return np.array(solutions)


class CrankDiffusion:
    def __init__(self, alpha=5.0, radius=1e-3, n=40):
        self.n = n
        self.alpha = alpha
        self.radius = radius
        self.objective = np.vectorize(self._objective, excluded="self")

    def alpha_prime(self, Kp):
        return self.alpha / Kp

    def solutions(self, Kp):
        return CrankEquationSolver(self.alpha_prime(Kp)).compute(self.n)

    def term(self, t, qn, Kp, D):
        return np.exp(-(D * t * qn**2) / (self.radius**2)) / (
            9 * (self.alpha_prime(Kp) + 1) + (self.alpha_prime(Kp) ** 2) * qn**2
        )

    def sum(self, t, Kp, D):
        return np.sum([self.term(t, qn, Kp, D) for qn in self.solutions(Kp)])

    def _objective(self, t, Kp, D):
        return self.alpha_prime(Kp) / (1 + self.alpha_prime(Kp)) + 6 * self.alpha_prime(
            Kp
        ) * self.sum(t, Kp, D)


class CrankDiffusionFitSolver(FitSolverInterface):
    _helper = CrankDiffusion()

    def __init__(self, alpha=3.9, radius=1.9e-3, **kwargs):
        self._helper.alpha = alpha
        self._helper.radius = radius
        super().__init__(**kwargs)

    @staticmethod
    def model(x, Kp, D):
        """
        Solve the modified Crank diffusion problem for a solid sphere

        .. math::

            \\gamma(t) = \\frac{C(t,R)}{C(0,R)} = \\frac{\\alpha}{1 + \\alpha} + 6 \\alpha \\sum\\limits_{i=1}^\\infty \\frac{\\exp \\left(-\\mathcal{D}\\frac{q_n^2 t}{R^2}\\right)}{9(\\alpha + 1) + \\alpha^2q_n^2}

        Where:

        .. math::

            \\forall q_n \\in \\mathbb{R}^+_0 \\, | \\, \\tan(q_n) = \\frac{3 q_n}{3 + \\alpha q_n^2}

        And:

        .. math::

            \\alpha = \\frac{V_l}{V_s} \\rightarrow \\alpha' = \\frac{V_l}{V_s K_p}

        :param x: Experimental time
        :param Kp: Partition constant
        :param D: Diffusion constant
        :return: Percentage of concentration saturation :math:`\gamma` at the surface of the sphere
        """
        return CrankDiffusionFitSolver._helper.objective(x[:, 0], Kp, D)


class RaneyKetonDehydrogenationFitSolver(FitSolverInterface):
    def __init__(self, n0=3.5, V=200e-6, **kwargs):
        self.n0 = n0  # m3 of isopropanol
        self.V = V  # mol
        super().__init__(**kwargs)

    def model(self, x, k1, k2):
        """

        .. math::

            r = k_1 \\theta_A = k_1\\frac{aA}{1 + aA + bB + cC}

        .. math::

            r = \\frac{1}{V}\\frac{\\mathrm{d}\\xi}{\\mathrm{d}t} \\simeq k_1\\frac{aA}{aA + bB} = k\\frac{a(n_0 - \\xi)}{a(n_0 - \\xi) + b\\xi}

        .. math::

            \\int\\limits_0^\\xi\\left(1 + \\frac{b\\xi}{a(n_0-\\xi)}\\right)\\mathrm{d}\\xi = \\int\\limits_0^t k_1V\\mathrm{d}t

        .. math::

            \\left(1 - \\frac{b}{a}\\right)\\xi - n_0\\frac{b}{a}\\ln\\left|\\frac{n_0 - \\xi}{n_0}\\right| = k_1Vt

        :param x:
        :param k1:
        :param k2:
        :return:
        """
        return (1 - k2) / k1 * (x[:, 0] / self.V) - (k2 / k1) * (
            self.n0 / self.V
        ) * np.log((self.n0 - x[:, 0]) / self.n0)
