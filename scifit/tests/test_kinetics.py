import os
import functools
import pathlib
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate

from scifit.interfaces.kinetics import KineticSolverInterface


print_report = bool(int(os.getenv("TESTS_PRINT_REPORT", 0)))


class GenericKineticTest:
    root_path = ".cache/media/tests/"
    format = "png"

    factory = KineticSolverInterface

    substance_index = 0
    substance_indices = None
    unsteady = None
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
            unsteady=self.unsteady,
        )

    def test_solve(self):
        solution = self.solver.integrate(t=self.t)

    def test_solve_against_model(self):
        solution = self.solver.integrate(self.t)
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
        self.assertTrue(
            np.all(
                np.isclose(solution.y, check.y, rtol=1e-4, atol=1e-6, equal_nan=True)
            )
        )

    def test_reactant_references(self):
        references = self.solver.references

    def test_model_formulas(self):
        formulas = self.solver.model_formulas()

    def test_model_formulas(self):
        equations = self.solver.model_equations()

    def test_conversion_ratio(self):
        self.solver.integrate(t=self.t)
        ratio = self.solver.convertion_ratio()

    def test_kinetic_rates(self):
        self.solver.integrate(t=self.t)
        ref = self.solver.rates()
        exp = self.solver.derivative()
        data = pd.concat([pd.DataFrame(ref), pd.DataFrame(exp)], axis=1)
        name = self.__class__.__name__
        # data.to_excel("{}/{}_check_rates.xlsx".format(self.media_path, name))
        # print(np.max(np.abs(ref -exp)))
        self.assertTrue(np.allclose(ref, exp, rtol=1e-2, atol=1e-3, equal_nan=True))

    # def test_kinetic_accelerations(self):
    #     self.solver.integrate(t=self.t)
    #     ref = self.solver.accelerations()
    #     exp = self.solver.derivative(derivative_order=2)
    #     data = pd.concat([pd.DataFrame(ref), pd.DataFrame(exp)], axis=1)
    #     name = self.__class__.__name__
    #     #data.to_excel("{}/{}_check_accelerations.xlsx".format(self.media_path, name))
    #     self.assertTrue(np.allclose(ref, exp, 1e-3))

    def test_first_derivative(self):
        self.solver.integrate(t=self.t)
        dxdt = self.solver.derivative(derivative_order=1)

    def test_second_derivative(self):
        self.solver.integrate(t=self.t)
        d2xdt2 = self.solver.derivative(derivative_order=2)

    def test_selectivities(self):
        self.solver.integrate(t=self.t)
        selectivities = self.solver.selectivities()

    def test_levenspiel(self):
        self.solver.integrate(t=self.t)
        L = self.solver.integrated_levenspiel()

    def test_dataset(self):
        name = self.__class__.__name__
        self.solver.integrate(t=self.t)
        data = self.solver.dataset()
        data.to_csv("{}/{}_data.csv".format(self.media_path, name), sep=";")

    def test_plot_solve(self):
        name = self.__class__.__name__
        self.solver.integrate(t=self.t)
        axe = self.solver.plot_solve(substance_indices=self.substance_indices)
        axe.figure.savefig("{}/{}_solve.{}".format(self.media_path, name, self.format))
        plt.close(axe.figure)

    def test_plot_solve_ratio(self):
        name = self.__class__.__name__
        self.solver.integrate(t=self.t)
        axe = self.solver.plot_solve_ratio(substance_indices=self.substance_indices)
        axe.figure.savefig(
            "{}/{}_solve_ratio.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def test_plot_rates(self):
        name = self.__class__.__name__
        self.solver.integrate(t=self.t)
        axe = self.solver.plot_rates(substance_indices=self.substance_indices)
        axe.figure.savefig("{}/{}_rates.{}".format(self.media_path, name, self.format))
        plt.close(axe.figure)

    def test_plot_accelerations(self):
        name = self.__class__.__name__
        self.solver.integrate(t=self.t)
        axe = self.solver.plot_accelerations(substance_indices=self.substance_indices)
        axe.figure.savefig(
            "{}/{}_accelerations.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def test_plot_selectivities(self):
        name = self.__class__.__name__
        self.solver.integrate(t=self.t)
        axe = self.solver.plot_selectivities(substance_indices=self.substance_indices)
        axe.figure.savefig(
            "{}/{}_selectivities.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def test_plot_global_selectivities(self):
        name = self.__class__.__name__
        self.solver.integrate(t=self.t)
        axe = self.solver.plot_global_selectivities(
            substance_indices=self.substance_indices
        )
        axe.figure.savefig(
            "{}/{}_selectivities_global.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def test_yields(self):
        name = self.__class__.__name__
        self.solver.integrate(t=self.t)
        axe = self.solver.plot_yields(substance_indices=self.substance_indices)
        axe.figure.savefig("{}/{}_yields.{}".format(self.media_path, name, self.format))
        plt.close(axe.figure)

    def test_plot_levenspiel(self):
        name = self.__class__.__name__
        self.solver.integrate(t=self.t)
        axe = self.solver.plot_levenspiel(substance_indices=self.substance_indices)
        axe.figure.savefig(
            "{}/{}_levenspiel.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def test_plot_integrated_levenspiel(self):
        name = self.__class__.__name__
        self.solver.integrate(t=self.t)
        axe = self.solver.plot_integrated_levenspiel(
            substance_indices=self.substance_indices
        )
        axe.figure.savefig(
            "{}/{}_levenspiel_integrated.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def test_plot_quotients(self):
        name = self.__class__.__name__
        self.solver.integrate(t=self.t)
        axe = self.solver.plot_quotients()
        axe.figure.savefig(
            "{}/{}_quotients.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def test_plot_quotient_rates(self):
        name = self.__class__.__name__
        self.solver.integrate(t=self.t)
        axe = self.solver.plot_quotient_rates()
        axe.figure.savefig(
            "{}/{}_quotient_rates.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def test_plot_quotient_rates(self):
        name = self.__class__.__name__
        self.solver.integrate(t=self.t)
        axe = self.solver.plot_quotient_rates()
        axe.figure.savefig(
            "{}/{}_quotient_rates.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def test_process_report(self):
        if print_report:
            name = self.__class__.__name__
            file = r"{}_report".format(name)
            self.solver.integrate(self.t)
            self.solver.report(
                file=file,
                path=self.media_path,
                mode="pdf",
                substance_indices=self.substance_indices,
            )


resolution = 5001


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
        return np.array(
            [
                -self.k0[0] * x[0],
                +self.k0[0] * x[0],
            ]
        )


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
        return np.array(
            [
                -self.k0[0] * x[0] * x[1],
                -self.k0[0] * x[0] * x[1],
                +self.k0[0] * x[0] * x[1],
            ]
        )


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
        return np.array(
            [
                +self.k0inv[0] * x[1],
                -self.k0inv[0] * x[1],
            ]
        )


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
        return np.array(
            [
                -self.k0[0] * x[0] + self.k0inv[0] * x[1],
                +self.k0[0] * x[0] - self.k0inv[0] * x[1],
            ]
        )


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
        return np.array(
            [
                -self.k0[0] * x[0] * x[1],
                +self.k0[0] * x[0] * x[1],
            ]
        )


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
        return np.array(
            [
                -self.k0[0] * x[0],
                +self.k0[0] * x[0] - self.k0[1] * x[1],
                +self.k0[1] * x[1],
            ]
        )


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
        return np.array(
            [
                -self.k0[0] * x[0],
                +self.k0[0] * x[0] - self.k0[1] * x[1],
                +self.k0[1] * x[1] - self.k0[2] * x[2],
                +self.k0[2] * x[2],
            ]
        )


class MultipleKinetics03(GenericKineticTest, TestCase):
    nur = np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, -1, 0, 1]])
    nup = None
    x0 = np.array([2e-3, 0.0, 0.0, 0.0])
    k0 = np.array([1e-2, 3e-2, 1e-1])
    t = np.linspace(0.0, 500.0, resolution)

    def model(self, t, x):
        return np.array(
            [
                -self.k0[0] * x[0],
                +self.k0[0] * x[0] - self.k0[1] * x[1] - self.k0[2] * x[1],
                +self.k0[1] * x[1],
                +self.k0[2] * x[1],
            ]
        )


class MultipleKinetics04(GenericKineticTest, TestCase):
    nur = np.array(
        [[-1, 1, 0, 0, 0], [0, -1, 1, 0, 0], [0, -1, 0, 1, 0], [0, 0, -1, -1, 1]]
    )
    nup = None
    x0 = np.array([2e-3, 0.0, 0.0, 0.0, 0.0])
    k0 = np.array([1e-2, 3e-2, 1e-1, 2e-1])
    t = np.linspace(0.0, 500.0, resolution)

    def model(self, t, x):
        return np.array(
            [
                -self.k0[0] * x[0],
                +self.k0[0] * x[0] - self.k0[1] * x[1] - self.k0[2] * x[1],
                +self.k0[1] * x[1] - self.k0[3] * x[2] * x[3],
                +self.k0[2] * x[1] - self.k0[3] * x[2] * x[3],
                +self.k0[3] * x[2] * x[3],
            ]
        )


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

    unsteady = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
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

    unsteady = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    substance_index = 2
    substance_indices = [4, 5]
    x0 = np.array([1, 3, 0.0, 0.0, 1, 1])
    k0 = np.array([1, 1, 1, 1])

    t = np.linspace(0.0, 50.0, resolution)

    def test_kinetic_rates(self):
        pass


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

    unsteady = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    substance_index = 2
    substance_indices = [4, 5]
    x0 = np.array([1, 1.7, 0.0, 0.0, 1, 1])
    k0 = np.array([1, 1, 1, 1])

    t = np.linspace(0.0, 50.0, resolution)


class MultipleKinetics08(GenericKineticTest, TestCase):
    nur = np.array(
        [
            [-1, -1, 0, 0],
            [-1, 0, -1, 0],
            [-1, 0, 0, -1],
        ]
    )
    nup = np.array(
        [
            [2, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
        ]
    )
    x0 = np.array([3e-3, 2e-3, 1e-3, 1e-9])
    k0 = np.array([2.1, 1, 9e-1])
    t = np.linspace(0.0, 2000.0, resolution)
    substance_index = 1

    def model(self, t, x):
        return np.array(
            [
                +self.k0[0] * x[0] * x[1]
                - self.k0[1] * x[0] * x[2]
                - self.k0[2] * x[0] * x[3],
                -self.k0[0] * x[0] * x[1] + self.k0[2] * x[0] * x[3],
                -self.k0[1] * x[0] * x[2],
                +self.k0[1] * x[0] * x[2] - self.k0[2] * x[0] * x[3],
            ]
        )
