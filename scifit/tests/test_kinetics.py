import pathlib
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scifit.interfaces.kinetics import KineticSolverInterface


class GenericASMKineticTest:
    root_path = ".cache/media/tests/"
    format = "png"

    factory = KineticSolverInterface

    nur = None
    nup = None
    x0 = None
    k0 = None
    k0inv = None
    mode = "direct"
    t = None

    def setUp(self):
        self.media_path = pathlib.Path(self.root_path) / format(
            self.factory.__module__.split(".")[-1]
        )
        self.media_path.mkdir(parents=True, exist_ok=True)
        self.solver = self.factory(
            self.nur, self.x0, self.k0, nup=self.nup, k0inv=self.k0inv, mode=self.mode
        )

    def test_solve(self):
        solution = self.solver.solve(t=self.t)

    def test_model_formula(self):
        formula = self.solver.model_formulas()

    def test_plot_solve(self):
        name = self.__class__.__name__
        self.solver.solve(t=self.t)
        axe = self.solver.plot_solve()
        axe.figure.savefig("{}/{}_solve.{}".format(self.media_path, name, self.format))
        plt.close(axe.figure)

    def test_plot_quotients(self):
        name = self.__class__.__name__
        self.solver.solve(t=self.t)
        axe = self.solver.plot_quotients()
        axe.figure.savefig("{}/{}_quotients.{}".format(self.media_path, name, self.format))
        plt.close(axe.figure)


class SimpleKinetic01(GenericASMKineticTest):
    nur = np.array([
        [-1, 1],
    ])
    nup = None
    x0 = np.array([2e-3, 0.0])
    k0 = np.array([1e-2])
    t = np.linspace(0.0, 500.0, 1000)


class SimpleKinetic01K0L0(SimpleKinetic01, TestCase):
    k0 = np.array([1e-2])


class SimpleKinetic01K0L1(SimpleKinetic01, TestCase):
    k0 = np.array([1e-1])


class SimpleKinetic01K0L2(SimpleKinetic01, TestCase):
    k0 = np.array([1e-0])


class SimpleKinetic02(GenericASMKineticTest):
    nur = np.array([
        [-1, -1, 1],
    ])
    nup = None
    x0 = np.array([2e-3, 5e-3, 0.0])
    k0 = np.array([1e-0])
    t = np.linspace(0.0, 500.0, 1000)


class SimpleKinetic02K0L0(SimpleKinetic02, TestCase):
    k0 = np.array([1e-0])


class SimpleKinetic02K0L1(SimpleKinetic02, TestCase):
    k0 = np.array([1e+1])


class SimpleKinetic02K0L2(SimpleKinetic02, TestCase):
    k0 = np.array([1e+2])


#
# class SimpleKinetic02(GenericASMKineticTest, TestCase):
#     nur = np.array([
#         [-1, 1, 0],
#         [0, -1, 1],
#     ])
#     nup = None
#     x0 = np.array([2e-3, 0.0, 0.0])
#     k0 = np.array([1e-2, 1e-3])
#     t = np.linspace(0.0, 3600.0, 500)
#
#
# class SimpleKinetic03(GenericASMKineticTest, TestCase):
#     nur = np.array([
#         [-1, 1, 0, 0],
#         [0, -1, 1, 0],
#         [0, 0, -1, 1],
#     ])
#     nup = None
#     x0 = np.array([2e-3, 0.0, 0.0, 0.0])
#     k0 = np.array([1e-2, 1e-3, 1e-1])
#     t = np.linspace(0.0, 3600.0, 500)
#
#
# class SimpleKinetic04(GenericASMKineticTest, TestCase):
#     nur = np.array([
#         [-1, -1, 1],
#     ])
#     nup = None
#     x0 = np.array([2e-3, 3e-3, 0.0])
#     k0 = np.array([1e+1])
#     t = np.linspace(0.0, 3600.0, 500)
#
#
# class SimpleKinetic05(GenericASMKineticTest, TestCase):
#     nur = np.array([
#         [-1, -2, 1],
#     ])
#     nup = None
#     x0 = np.array([2e-3, 3e-3, 0.0])
#     k0 = np.array([1e+4])
#     t = np.linspace(0.0, 3600.0, 500)
#
#
# class SimpleKinetic06(GenericASMKineticTest, TestCase):
#     nur = np.array([
#         [-1, 1, 0, 0, 0],
#         [0, -1, 1, 0, 0],
#         [0, -1, 0, 1, 0],
#         [0, 0, -1, -1, 1]
#     ])
#     nup = None
#     x0 = np.array([2e-3, 0.0, 0.0, 0.0, 0.0])
#     k0 = np.array([1e-2, 3e-2, 1e-1, 1e-2])
#     t = np.linspace(0.0, 500.0, 500)
#
#
# class SimpleKinetic07(GenericASMKineticTest, TestCase):
#     nur = np.array([
#         [-5, -1, 2,  2, 4, 0],
#         [-5, 1, -2, -2, 6, 5],
#     ])
#     nup = None
#     x0 = np.array([2e-1, 5.e-2, 0.0, 0.0, 0.0, 0.0])
#     k0 = np.array([9e+1, 1e+2])
#     t = np.linspace(0.0, 2000.0, 500)
#
#
# class SimpleKinetic08(GenericASMKineticTest, TestCase):
#     # https://en.wikipedia.org/wiki/Brusselator
#     nur = np.array([
#         [-1, 0, 0,  0, 0, 0],
#         [0, 0, 0, 0, -2, -1],
#         [0, -1, 0, 0, -1, 0],
#         [0, 0, 0, 0,  -1, 0],
#     ])
#     nup = np.array([
#         [0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 3, 0],
#         [0, 0, 1, 0, 0, 1],
#         [0, 0, 0, 1, 0, 0]
#     ])
#     x0 = np.array([1, 3, 1, 1, 1, 1])
#     k0 = np.array([1, 1, 1, 1])
#     t = np.linspace(0.0, 500.0, 500)
#
