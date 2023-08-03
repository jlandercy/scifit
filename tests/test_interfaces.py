"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`newproject.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np

from scifit.interfaces.generic import FitSolverInterface
from scifit.errors.base import *


class TestFitSolverInterfaceErrors(TestCase):

    def setUp(self) -> None:
        self.solver = FitSolverInterface()

    def test_bad_shaped_arrays(self):
        with self.assertRaises(InputDataError):
            solver = self.solver.store([[1], [2]], [3, 4, 5])
        with self.assertRaises(InputDataError):
            solver = self.solver.store([[1, 2, 3], [4, 5, 6]], [7, 8, 9])

    def test_numerical_data(self):
        with self.assertRaises(InputDataError):
            solver = self.solver.store([["1"]], [2])
        with self.assertRaises(InputDataError):
            solver = self.solver.store([["1"], ["2"]], [3, 4])
        with self.assertRaises(InputDataError):
            solver = self.solver.store([["1", "2", "3"], ["4", "5", "6"]], [7, 8])

    def test_missing_data(self):
        with self.assertRaises(InputDataError):
            solver = self.solver.store([[np.nan]], [np.nan])
        with self.assertRaises(InputDataError):
            solver = self.solver.store([[1], [float("nan")]], [3, 4])
        with self.assertRaises(InputDataError):
            solver = self.solver.store([[1, 2, 3], [4, 5, np.nan]], [5, 6])


class GenericTestFitSolverInterface:

    xdata = []
    ydata = []

    def setUp(self) -> None:
        self.solver = FitSolverInterface()
        self.solver.store(self.xdata, self.ydata)

    def test_missing_model(self):
        with self.assertRaises(MissingModel):
            self.solver.model([], [])

    def test_space_sizes(self):
        self.assertEqual(self.solver.observation_space_size, self.solver.n)
        self.assertEqual(self.solver.variable_space_size, self.solver.m)
        self.assertEqual(self.solver.parameter_space_size, self.solver.k)
        self.assertEqual(self.solver.n, self.solver._xdata.shape[0])
        self.assertEqual(self.solver.m, self.solver._xdata.shape[1])
        self.assertEqual(self.solver.k, len(self.solver.signature.parameters) - 1)

    def test_linear_space_generation(self):
        xlin = self.solver.space(mode="lin", xmin=-10., xmax=+10., resolution=200)
        self.assertIsInstance(xlin, np.ndarray)
        self.assertEqual(xlin.ndim, 1)
        self.assertEqual(xlin.shape[0], 200)
        self.assertEqual(xlin.min(), -10.)
        self.assertEqual(xlin.max(), +10.)


class TestFitSolverInterfaceWithListSingleton(GenericTestFitSolverInterface, TestCase):
    xdata = [[1]]
    ydata = [2]


class TestFitSolverInterfaceWithList1D(GenericTestFitSolverInterface, TestCase):
    xdata = [[1], [2], [3]]
    ydata = [4, 5, 6]


class TestFitSolverInterfaceWithList2D(GenericTestFitSolverInterface, TestCase):
    xdata = [[1, 2, 3], [4, 5, 6]]
    ydata = [7, 8]


class TestFitSolverInterfaceWithArraySingleton(GenericTestFitSolverInterface, TestCase):
    xdata = np.array([[1]])
    ydata = np.array([2])


class TestFitSolverInterfaceWithArray1D(GenericTestFitSolverInterface, TestCase):
    xdata = np.array([[1], [2], [3]])
    ydata = np.array([4, 5, 6])


class TestFitSolverInterfaceWithArray2D(GenericTestFitSolverInterface, TestCase):
    xdata = np.array([[1, 2, 3], [4, 5, 6]])
    ydata = np.array([7, 8])


class TestFitSolverInterfaceWithArray2DTypical(GenericTestFitSolverInterface, TestCase):
    xdata = np.random.randn(500).reshape(100, 5)
    ydata = np.random.randn(100)

