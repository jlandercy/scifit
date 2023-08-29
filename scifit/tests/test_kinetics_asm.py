import pathlib
from unittest import TestCase

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scifit.interfaces.kinetics import ActivatedStateModelKinetic


class GenericASMKineticTest:

    root_path = ".cache/media/tests/"
    format = "png"

    factory = ActivatedStateModelKinetic

    nus = None
    x0 = None
    k0 = None
    t = None

    def setUp(self):
        self.media_path = pathlib.Path(self.root_path) / format(
            self.factory.__module__.split(".")[-1]
        )
        self.media_path.mkdir(parents=True, exist_ok=True)
        self.solver = self.factory(self.nus, self.x0, self.k0)

    def test_solve(self):
        solution = self.solver.solve(t=self.t)

    def test_plot_solve(self):
        name = self.__class__.__name__
        self.solver.solve(t=self.t)
        axe = self.solver.plot_solve()
        axe.figure.savefig(
            "{}/{}_solve.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)


class SimpleKinetic01(GenericASMKineticTest, TestCase):
    nus = np.array([-1, 1])
    x0 = np.array([2e-3, 0.])
    k0 = 1e-3
    t = np.linspace(0., 3600., 500)


class SimpleKinetic02(GenericASMKineticTest, TestCase):
    nus = np.array([-1, -2, 1])
    x0 = np.array([2e-3, 8e-3, 0.])
    k0 = 1e+2
    t = np.linspace(0., 3600., 500)
