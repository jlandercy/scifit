import numpy as np
from scipy import integrate, optimize

from scifit.interfaces.generic import FitSolverInterface


class DebyeFitSolver(FitSolverInterface):
    @staticmethod
    def _integral(n):
        def integrand(t):
            return t**n / (np.exp(t) - 1)

        @np.vectorize
        def wrapped(x):
            return (n / np.power(x, n)) * integrate.quad(integrand, 0, x)[0]

        return wrapped

    @staticmethod
    def model(x, theta):
        return 3 * x[:, 0] * DebyeFitSolver._integral(3)(theta / x[:, 0])


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
    def __init__(self, alpha=5, radius=1e-3, n=10):
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
    _helper = CrankDiffusion(alpha=3.9, radius=1.9e-3)

    @staticmethod
    def model(x, Kp, D):
        return CrankDiffusionFitSolver._helper.objective(x[:, 0], Kp, D)
