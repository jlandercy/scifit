import pathlib
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scifit.interfaces.kinetics import ActivatedStateModelKinetic


class GenericASMKineticTest:
    root_path = ".cache/media/tests/"
    format = "png"

    factory = ActivatedStateModelKinetic

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
        formula = self.solver.model_formula()
        print(formula)

    def test_plot_solve(self):
        name = self.__class__.__name__
        self.solver.solve(t=self.t)
        axe = self.solver.plot_solve()
        axe.figure.savefig("{}/{}_solve.{}".format(self.media_path, name, self.format))
        plt.close(axe.figure)


class SimpleKinetic01(GenericASMKineticTest, TestCase):
    nur = np.array([
        [-1, 1, 0, 0],
        [0, -1, 1, 0],
        [0, 0, -1, 1],
    ])
    nup = None
    x0 = np.array([2e-3, 0.0, 0.0, 0.0])
    k0 = np.array([1e-2, 1e-2, 1e-2])
    t = np.linspace(0.0, 1200.0, 250)

#
# class SimpleKinetic01QuickNotation(GenericASMKineticTest, TestCase):
#     nur = np.array([[-1, 1]])
#     nup = None
#     x0 = np.array([2e-3, 0.0])
#     k0 = np.array([1e-3])
#     t = np.linspace(0.0, 3600.0, 500)
#
#
# class SimpleKinetic02(GenericASMKineticTest, TestCase):
#     nur = np.array([[-1, -2, 0]])
#     nup = np.array([[0, 0, 1]])
#     x0 = np.array([2e-3, 8e-3, 0.0])
#     k0 = np.array([1e2])
#     t = np.linspace(0.0, 3600.0, 500)
#
#
# class SimpleKinetic02QuickNotation(GenericASMKineticTest, TestCase):
#     nur = np.array([[-1, -2, 1]])
#     nup = None
#     x0 = np.array([2e-3, 8e-3, 0.0])
#     k0 = np.array([1e2])
#     t = np.linspace(0.0, 3600.0, 500)
#
#
# class SimpleKinetic03(GenericASMKineticTest, TestCase):
#     nur = np.array([[-1, -2, 0, 0]])
#     nup = np.array([[0, 0, 1, 2]])
#     x0 = np.array([2e-3, 8e-3, 0.0, 0.0])
#     k0 = np.array([1e2])
#     t = np.linspace(0.0, 3600.0, 500)
#
#
# class SimpleKinetic03QuickNotation(GenericASMKineticTest, TestCase):
#     nur = np.array([[-1, -2, 1, 2]])
#     nup = None
#     x0 = np.array([2e-3, 8e-3, 0.0, 0.0])
#     k0 = np.array([1e2])
#     t = np.linspace(0.0, 3600.0, 500)
#
#
# class SimpleKinetic04(GenericASMKineticTest, TestCase):
#     nur = np.array([[-2, -1, 0]])
#     nup = np.array([[1, 0, 1]])
#     x0 = np.array([2e-3, 8e-3, 0.0])
#     k0 = np.array([1e2])
#     t = np.linspace(0.0, 3600.0, 500)
