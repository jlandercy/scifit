import pathlib
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate

from scifit.interfaces.kinetics import KineticSolverInterface


class GenericKineticTest:
    root_path = ".cache/media/tests/"
    format = "png"

    factory = KineticSolverInterface

    substance_index = 0
    substance_indices = None
    steady = None
    nur = None
    nup = None
    x0 = None
    k0 = None
    k0inv = None
    mode = "direct"
    t = None

    @staticmethod
    def model(t, x):
        raise NotImplemented("Model not defined")

    def setUp(self):
        self.media_path = pathlib.Path(self.root_path) / format(
            self.factory.__module__.split(".")[-1]
        )
        self.media_path.mkdir(parents=True, exist_ok=True)
        self.solver = self.factory(
            self.nur,
            self.x0,
            self.k0,
            nup=self.nup,
            k0inv=self.k0inv,
            mode=self.mode,
            substance_index=self.substance_index,
            steady=self.steady,
        )

    def test_solve(self):
        solution = self.solver.fit(t=self.t)

    def test_solve_against_model(self):
        solution = self.solver.fit(self.t)
        check = integrate.solve_ivp(
            self.model,
            [self.t.min(), self.t.max()],
            self.x0,
            t_eval=self.t,
            dense_output=True,
            method="LSODA",
            min_step=1e-8,
            atol=1e-12,
            rtol=1e-10,
        )
        self.assertTrue(np.all(np.isclose(solution.t, check.t)))
        self.assertTrue(np.all(np.isclose(solution.y, check.y)))

    def test_reactant_references(self):
        references = self.solver.references

    def test_model_formula(self):
        formula = self.solver.model_formulas()

    def test_conversion_ratio(self):
        self.solver.fit(t=self.t)
        ratio = self.solver.convertion_ratio()

    def test_first_derivative(self):
        self.solver.fit(t=self.t)
        dxdt = self.solver.derivative(derivative_order=1)

    def test_second_derivative(self):
        self.solver.fit(t=self.t)
        d2xdt2 = self.solver.derivative(derivative_order=2)

    def test_selectivities(self):
        self.solver.fit(t=self.t)
        selectivities = self.solver.selectivities()

    def test_levenspiel(self):
        self.solver.fit(t=self.t)
        L = self.solver.levenspiel()

    def test_dataset(self):
        name = self.__class__.__name__
        self.solver.fit(t=self.t)
        data = self.solver.dataset()
        data.to_csv("{}/{}_data.csv".format(self.media_path, name), sep=";")

    def test_plot_solve(self):
        name = self.__class__.__name__
        self.solver.fit(t=self.t)
        axe = self.solver.plot_solve(substance_indices=self.substance_indices)
        axe.figure.savefig("{}/{}_solve.{}".format(self.media_path, name, self.format))
        plt.close(axe.figure)

    def test_plot_solve_ratio(self):
        name = self.__class__.__name__
        self.solver.fit(t=self.t)
        axe = self.solver.plot_solve_ratio(substance_indices=self.substance_indices)
        axe.figure.savefig(
            "{}/{}_solve_ratio.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def test_plot_first_derivative(self):
        name = self.__class__.__name__
        self.solver.fit(t=self.t)
        axe = self.solver.plot_first_derivative(substance_indices=self.substance_indices)
        axe.figure.savefig("{}/{}_dxdt.{}".format(self.media_path, name, self.format))
        plt.close(axe.figure)

    def test_plot_second_derivative(self):
        name = self.__class__.__name__
        self.solver.fit(t=self.t)
        axe = self.solver.plot_second_derivative(substance_indices=self.substance_indices)
        axe.figure.savefig("{}/{}_d2xdt2.{}".format(self.media_path, name, self.format))
        plt.close(axe.figure)

    def test_plot_selectivities(self):
        name = self.__class__.__name__
        self.solver.fit(t=self.t)
        axe = self.solver.plot_selectivities(substance_indices=self.substance_indices)
        axe.figure.savefig(
            "{}/{}_selectivities.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def test_plot_levenspiel(self):
        name = self.__class__.__name__
        self.solver.fit(t=self.t)
        axe = self.solver.plot_levenspiel(substance_indices=self.substance_indices)
        axe.figure.savefig(
            "{}/{}_levenspiel.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def test_plot_quotients(self):
        name = self.__class__.__name__
        self.solver.fit(t=self.t)
        axe = self.solver.plot_quotients()
        axe.figure.savefig(
            "{}/{}_quotients.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)


resolution = 5000


class SimpleKinetic01(GenericKineticTest):
    """
    A -> B
    """

    nur = np.array(
        [
            [-1, 1],
        ]
    )
    nup = None
    x0 = np.array([2e-3, 0.0])
    k0 = np.array([1e-1])
    k0inv = np.array([1e-4])
    t = np.linspace(0.0, 500.0, resolution)

    def model(self, t, x):
        return np.array([
            -self.k0[0] * x[0],
            +self.k0[0] * x[0],
        ])


class SimpleKinetic01K0L0(SimpleKinetic01, TestCase):
    k0 = np.array([1e-2])


class SimpleKinetic01K0L1(SimpleKinetic01, TestCase):
    k0 = np.array([1e-1])


class SimpleKinetic01K0L2(SimpleKinetic01, TestCase):
    k0 = np.array([1e0])


class SimpleKinetic02(GenericKineticTest):
    """
    A + B -> C
    """

    nur = np.array(
        [
            [-1, -1, 1],
        ]
    )
    nup = None
    x0 = np.array([2e-3, 5e-3, 0.0])
    k0 = np.array([1e-0])
    t = np.linspace(0.0, 500.0, resolution)

    def model(self, t, x):
        return np.array([
            -self.k0[0] * x[0] * x[1],
            -self.k0[0] * x[0] * x[1],
            +self.k0[0] * x[0] * x[1],
        ])


class SimpleKinetic02K0L0(SimpleKinetic02, TestCase):
    k0 = np.array([1e-0])


class SimpleKinetic02K0L1(SimpleKinetic02, TestCase):
    k0 = np.array([1e1])


class SimpleKinetic02K0L2(SimpleKinetic02, TestCase):
    k0 = np.array([1e2])


class SimpleKinetic03(GenericKineticTest):
    """
    A <- B
    """

    nur = np.array(
        [
            [-1, 1],
        ]
    )
    substance_index = 1
    nup = None
    mode = "indirect"
    x0 = np.array([0.0, 2e-3])
    k0 = np.array([1e-2])
    k0inv = np.array([1e-2])
    t = np.linspace(0.0, 500.0, resolution)

    def model(self, t, x):
        return np.array([
            +self.k0inv[0] * x[1],
            -self.k0inv[0] * x[1],
        ])


class SimpleKinetic03K0L0(SimpleKinetic03, TestCase):
    k0inv = np.array([1e-2])


class SimpleKinetic03K0L1(SimpleKinetic03, TestCase):
    k0inv = np.array([1e-1])


class SimpleKinetic03K0L2(SimpleKinetic03, TestCase):
    k0inv = np.array([1e0])


class SimpleKinetic04(GenericKineticTest):
    """
    A <=> B
    """

    nur = np.array(
        [
            [-1, 1],
        ]
    )
    nup = None
    mode = "equilibrium"
    x0 = np.array([3e-3, 1e-3])
    k0 = np.array([1e-2])
    k0inv = np.array([1e-4])
    t = np.linspace(0.0, 500.0, resolution)

    def model(self, t, x):
        return np.array([
            -self.k0[0] * x[0] +self.k0inv[0] * x[1],
            +self.k0[0] * x[0] -self.k0inv[0] * x[1],
        ])


class SimpleKinetic04K0L0(SimpleKinetic04, TestCase):
    k0 = np.array([1e-3])


class SimpleKinetic04K0L1(SimpleKinetic04, TestCase):
    k0 = np.array([1e-2])


class SimpleKinetic04K0L2(SimpleKinetic04, TestCase):
    k0 = np.array([1e-1])


class SimpleKinetic05(GenericKineticTest):
    """
    A + B -> 2B
    """

    nur = np.array(
        [
            [-1, -1],
        ]
    )
    nup = np.array(
        [
            [0, 2],
        ]
    )
    x0 = np.array([5e-3, 2e-3])
    k0 = np.array([1e-1])
    t = np.linspace(0.0, 500.0, resolution)

    def model(self, t, x):
        return np.array([
            -self.k0[0] * x[0] * x[1],
            +self.k0[0] * x[0] * x[1],
        ])


class SimpleKinetic05K0L0(SimpleKinetic05, TestCase):
    k0 = np.array([1e-1])


class SimpleKinetic05K0L1(SimpleKinetic05, TestCase):
    k0 = np.array([1e0])


class SimpleKinetic05K0L2(SimpleKinetic05, TestCase):
    k0 = np.array([1e1])


class MultipleKinetics01(GenericKineticTest, TestCase):
    nur = np.array(
        [
            [-1, 1, 0],
            [0, -1, 1],
        ]
    )
    nup = None
    x0 = np.array([2e-3, 0.0, 0.0])
    k0 = np.array([1e-2, 1e-3])
    t = np.linspace(0.0, 500.0, resolution)

    def model(self, t, x):
        return np.array([
            -self.k0[0] * x[0],
            +self.k0[0] * x[0] - self.k0[1] * x[1],
            +self.k0[1] * x[1],
        ])


class MultipleKinetics02(GenericKineticTest, TestCase):
    nur = np.array(
        [
            [-1, 1, 0, 0],
            [0, -1, 1, 0],
            [0, 0, -1, 1],
        ]
    )
    nup = None
    x0 = np.array([2e-3, 0.0, 0.0, 0.0])
    k0 = np.array([1e-2, 1e-3, 1e-1])
    t = np.linspace(0.0, 500.0, resolution)

    def model(self, t, x):
        return np.array([
            -self.k0[0] * x[0],
            +self.k0[0] * x[0] - self.k0[1] * x[1],
            +self.k0[1] * x[1] - self.k0[2] * x[2],
            +self.k0[2] * x[2],
        ])


class MultipleKinetics03(GenericKineticTest, TestCase):
    nur = np.array([
        [-1, 1, 0, 0],
        [0, -1, 1, 0],
        [0, -1, 0, 1]]
    )
    nup = None
    x0 = np.array([2e-3, 0.0, 0.0, 0.0])
    k0 = np.array([1e-2, 3e-2, 1e-1])
    t = np.linspace(0.0, 500.0, resolution)

    def model(self, t, x):
        return np.array([
            -self.k0[0] * x[0],
            +self.k0[0] * x[0] - self.k0[1] * x[1] - self.k0[2] * x[1],
            +self.k0[1] * x[1],
            +self.k0[2] * x[1],
        ])


class MultipleKinetics04(GenericKineticTest, TestCase):
    nur = np.array([
        [-1, 1, 0, 0, 0],
        [0, -1, 1, 0, 0],
        [0, -1, 0, 1, 0],
        [0, 0, -1, -1, 1]
    ])
    nup = None
    x0 = np.array([2e-3, 0.0, 0.0, 0.0, 0.0])
    k0 = np.array([1e-2, 3e-2, 1e-1, 2e-1])
    t = np.linspace(0.0, 500.0, resolution)

    def model(self, t, x):
        return np.array([
            -self.k0[0] * x[0],
            +self.k0[0] * x[0] - self.k0[1] * x[1] - self.k0[2] * x[1],
            +self.k0[1] * x[1] - self.k0[3] * x[2] * x[3],
            +self.k0[2] * x[1] - self.k0[3] * x[2] * x[3],
            +self.k0[3] * x[2] * x[3],
        ])


class MultipleKinetics05(GenericKineticTest, TestCase):
    """
    Brusselator (batch)
    """
    nur = np.array(
        [
            [-1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -2, -1],
            [0, -1, 0, 0, -1, 0],
            [0, 0, 0, 0, -1, 0],
        ]
    )
    nup = np.array(
        [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 3, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
        ]
    )

    def model(self, t, x):
        return np.array(
            [
                -self.k0[0] * x[0],  # A
                -self.k0[2] * x[1] * x[4],  # B
                +self.k0[2] * x[1] * x[4],  # D
                +self.k0[3] * x[4],  # E
                +self.k0[0] * x[0]
                + self.k0[1] * x[4] ** 2 * x[5]
                - self.k0[2] * x[1] * x[4]
                - self.k0[3] * x[4],  # X
                -self.k0[1] * x[4] ** 2 * x[5] + self.k0[2] * x[1] * x[4],  # Y
            ]
        )

    steady = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    substance_indices = None
    x0 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    # k0 = np.array([1e-3, 2e-1, 1e-2, 1e-3])  # 1 oscillation
    # k0 = np.array([1e-3, 1e-2, 5e-2, 1e-4])  # limit oscillation

    # k0 = np.array([5e-4, 1e-1, 5e-2, 1e-4]) # slow consumption - 10/10
    # k0 = np.array([1, 100, 10, 1.3])
    k0 = np.array([0.6, 84, 6.5, 1.7])

    t = np.linspace(0.0, 10.0, resolution)


class MultipleKinetics06(MultipleKinetics05, TestCase):
    """
    Brusselator (steady)
    """
    def model(self, t, x):
        return np.array(
            [
                0.0,  # A
                0.0,  # B
                +self.k0[2] * x[1] * x[4],  # D
                +self.k0[3] * x[4],  # E
                +self.k0[0] * x[0]
                + self.k0[1] * x[4] ** 2 * x[5]
                - self.k0[2] * x[1] * x[4]
                - self.k0[3] * x[4],  # X
                -self.k0[1] * x[4] ** 2 * x[5] + self.k0[2] * x[1] * x[4],  # Y
            ]
        )

    steady = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    substance_indices = [4, 5]
    x0 = np.array([1, 3, 0.0, 0.0, 1, 1])
    k0 = np.array([1, 1, 1, 1])

    t = np.linspace(0.0, 50.0, resolution)


class MultipleKinetics07(MultipleKinetics06, TestCase):
    """
    Brusselator (steady)
    """
    def model(self, t, x):
        return np.array(
            [
                0.0,  # A
                0.0,  # B
                +self.k0[2] * x[1] * x[4],  # D
                +self.k0[3] * x[4],  # E
                +self.k0[0] * x[0]
                + self.k0[1] * x[4] ** 2 * x[5]
                - self.k0[2] * x[1] * x[4]
                - self.k0[3] * x[4],  # X
                -self.k0[1] * x[4] ** 2 * x[5] + self.k0[2] * x[1] * x[4],  # Y
            ]
        )

    steady = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    substance_indices = [4, 5]
    x0 = np.array([1, 1.7, 0.0, 0.0, 1, 1])
    k0 = np.array([1, 1, 1, 1])

    t = np.linspace(0.0, 50.0, resolution)
