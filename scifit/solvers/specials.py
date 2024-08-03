import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize

from scifit.interfaces.solvers import FitSolver1D


class DebyeInternalEnergyFitSolver(FitSolver1D):
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


class CrankDiffusionFitSolver(FitSolver1D):
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


class RaneyKetonDehydrogenationFitSolver(FitSolver1D):
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


class AcidBasePartition:
    """

    .. math::

        \\alpha_i = \\frac{M_i(H)}{P(H)} = \\frac{H^{n-i} \\prod\\limits_{j=0}^{i} Ka_i}{\\sum\\limits_{i=0}^n H^{n-i} \\prod\\limits_{j=0}^{i} Ka_i}

    """

    def __init__(self, pKa):
        self.pKa = np.array(pKa)

    @property
    def Ka(self):
        return self.pinv(self.pKa)

    @property
    def n(self):
        return len(self.pKa)

    @staticmethod
    def p(x):
        return -np.log10(x)

    @staticmethod
    def pinv(x):
        return np.power(10., -x)

    def monom(self, i, pH):
        return np.power(self.pinv(pH), i) * np.prod(self.Ka[:(self.n - i)])

    def polynom(self, pH):
        return np.sum([self.monom(i, pH) for i in range(self.n + 1)], axis=0)

    def polynom_coefficients(self):
        return [np.prod(self.Ka[:i]) for i in range(self.n + 1)]

    def polynom_roots(self):
        return np.roots(self.polynom_coefficients())

    # def polynom(self, pH):
    #    return np.poly1d(self.polynom_coefficients())(self.pinv(pH))

    def alpha(self, i, pH):
        return self.monom(i, pH) / self.polynom(pH)

    def alphas(self, pH):

        a = np.array([
            self.alpha(self.n - i, pH)
            for i in range(self.n + 1)
        ]).T

        if not np.allclose(np.sum(a.T, axis=0), 1.):
            raise ValueError("Partition functions does not sum up to unity")

        return a

    def equilibrium_system(self, C, Ct, Na):
        return np.array([
            # Acid/Base Equilibria:
            (C[i + 1] * C[-1]) - C[i] * self.Ka[i]
            for i in range(self.n)
        ] + [
            # Charge Balance:
            np.sum(np.arange(self.n + 1) * C[:-1]) - Na,
            # Mass Balance:
            np.sum(C[:-1]) - Ct
        ])

    def equilibrium(self, C):

        Ct = np.sum(C[:-1])
        Na = np.sum(np.arange(self.n + 1) * C[:-1])

        sol, info, code, message = optimize.fsolve(
            self.equilibrium_system,
            x0=np.array([1.] * (self.n + 2)),
            args=(Ct, Na),
            full_output=True,
            #xtol=1e-9,
        )

        # if code != 1:
        #     raise ValueError("Solution not found: %s" % message)

        check = self.equilibrium_system(sol, Ct, Na)
        if not np.allclose(check, 0.):
            raise ValueError("Roots not enough small: %s" % check)

        return sol

    def plot(self, pHmin=0, pHmax=14, resolution=500):

        pH = np.linspace(pHmin, pHmax, resolution)
        alphas = self.alphas(pH)

        fig, axe = plt.subplots()
        axe.plot(pH, alphas)

        axe.set_title("Acid/Base Partitions pKa=%s" % self.pKa)
        axe.set_xlabel(r"Proton concentration, $pH$ [-]")
        axe.set_ylabel(r"Partition Functions, $\alpha_i$ [-]")
        axe.legend([r"$\alpha_{%d}$" % i for i in reversed(range(self.n + 1))])
        axe.grid()

        return axe

    def plot_solution(self, C, axe=None):

        if axe is None:
            axe = self.plot()

        pH = self.p(C[-1])
        alphas = C[:-1] / np.sum(C[:-1])

        axe.axvline(pH, linestyle="--", color="black")
        axe.scatter([pH] * (self.n + 1), alphas, color="black")

        return axe

    def plot_monoms(self, pHmin=0, pHmax=14, resolution=500):

        pH = np.linspace(pHmin, pHmax, resolution)
        D = self.polynom(pH)

        fig, axe = plt.subplots()
        for i in reversed(range(self.n + 1)):
            axe.semilogy(pH, self.monom(i, pH), label=r"$N_{%d}$" % i)
        axe.semilogy(pH, D, "--", color="black", linewidth=3, label="$D = \sum N_i$")

        axe.set_title("Partition monoms")
        axe.set_xlabel("Proton concentration, $pH$ [-]")
        axe.set_ylabel("Monoms, $N_i(pH)$ [-]")
        axe.legend()
        axe.grid()

        return axe

    def plot_validity(self, pHmin=0, pHmax=14, resolution=500):

        pH = np.linspace(pHmin, pHmax, resolution)
        C0 = []
        for x in pH:
            a0 = self.alphas(x)
            try:
                C0.append(self.equilibrium(list(a0) + [1e-7]))
            except:
                C0.append([np.nan] * (self.n + 2))
        C0 = np.array(C0)

        fig, axe = plt.subplots()
        axe.plot(pH, pH, label="Baseline")
        axe.plot(pH, -np.log10(C0[:, -1]), label="Estimation")

        axe.set_title("Solution agreement")
        axe.set_xlabel("Proton concentration, $pH$ [-]")
        axe.set_ylabel("Proton concentration, $pH$ [-]")
        axe.legend()
        axe.grid()

        return axe


class ComplexPartition:
    """

    .. math::

        \\alpha_i = \\frac{M_i(L)}{P(L)} = \\frac{L^{i} \\prod\\limits_{j=0}^{i} k_i}{\\sum\\limits_{i=0}^n L^{i} \\prod\\limits_{j=0}^{i} k_i}

    """

    def __init__(self, pK):
        self.pK = np.array(pK)

    @property
    def K(self):
        return np.power(10., self.pK)

    @property
    def n(self):
        return len(self.pK)

    def monom(self, i, L):
        return np.power(L, i) * np.prod(self.K[:i])

    def polynom(self, L):
        return np.sum([self.monom(i, L) for i in range(self.n + 1)], axis=0)

    def alpha(self, i, L):
        return self.monom(i, L) / self.polynom(L)

    def alphas(self, L):

        a = np.array([
            self.alpha(i, L)
            for i in range(self.n + 1)
        ]).T

        if not np.allclose(np.sum(a.T, axis=0), 1.):
            raise ValueError("Partition functions does not sum up to unity")

        return a

    def plot(self, pLmin=-5, pLmax=2, resolution=500):

        L = np.logspace(pLmin, pLmax, resolution)
        As = self.alphas(L)

        fig, axe = plt.subplots()
        axe.semilogx(L, As)
        axe.set_title("Complex Partition Functions")
        axe.set_xlabel("Ligand Concentration, L [mol/L]")
        axe.set_ylabel(r"Partition Function, $\alpha_i$ [-]")
        axe.legend([r"$\alpha_{%d}$" % i for i in range(self.n + 1)])
        axe.grid()

        return axe
