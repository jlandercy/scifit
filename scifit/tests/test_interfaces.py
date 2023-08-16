from unittest import TestCase

import numpy as np

from scifit.errors.base import *
from scifit.interfaces.generic import FitSolverInterface
from scifit.tests.helpers import GenericTestFitSolverInterface


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
