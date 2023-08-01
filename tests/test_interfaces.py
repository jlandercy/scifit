"""
Module :mod:`scifit.tests.test_interfaces` implements test suite for the
class :class:`newproject.interfaces.generic.GenericInterface` and its children.
"""

from unittest import TestCase

import numpy as np

from scifit.interfaces.generic import FitSolverInterface
from scifit.errors.base import *


class TestFitSolverInterface(TestCase):

    def test_missing_model(self):
        solver = FitSolverInterface([], [])
        with self.assertRaises(MissingModel):
            solver.model([], [])

    def test_valid_lists(self):
        solver = FitSolverInterface([], [])
        solver = FitSolverInterface([1, 2], [3, 4])
        solver = FitSolverInterface([[1, 2, 3], [4, 5, 6]], [3, 4])

    def test_valid_arrays(self):
        solver = FitSolverInterface(np.array([]), np.array([]))
        solver = FitSolverInterface(np.array([1, 2]), np.array([3, 4]))
        solver = FitSolverInterface(np.array([[1, 2, 3], [4, 5, 6]]), np.array([3, 4]))

    def test_bad_shaped_arrays(self):
        with self.assertRaises(InputDataError):
            solver = FitSolverInterface([], [1])
        with self.assertRaises(InputDataError):
            solver = FitSolverInterface([1, 2], [3, 4, 1])
        with self.assertRaises(InputDataError):
            solver = FitSolverInterface([[1, 2, 3], [4, 5, 6]], [3, 4, 1])

    def test_numerical_data(self):
        with self.assertRaises(InputDataError):
            solver = FitSolverInterface(["1"], [1])
        with self.assertRaises(InputDataError):
            solver = FitSolverInterface(["y", "x"], [3, 4])
        with self.assertRaises(InputDataError):
            solver = FitSolverInterface([["1", "2", "3"], ["4", "5", "y"]], [3, 4])

    def test_missing_data(self):
        with self.assertRaises(InputDataError):
            solver = FitSolverInterface([np.nan], [np.nan])
        with self.assertRaises(InputDataError):
            solver = FitSolverInterface([1, float("nan")], [3, 4])
        with self.assertRaises(InputDataError):
            solver = FitSolverInterface([[1, 2, 3], [4, 5, np.nan]], [3, 4])
