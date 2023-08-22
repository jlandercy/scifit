"""
Unit tests fixed issues are:
 - Bug detected during tests
 - Unit test written from MCVE
 - Decision, correction/fix
"""

from unittest import TestCase

import numpy as np
import pandas as pd

from scifit.solvers import scientific
from scifit.tests.helpers import GenericTestFitSolver
from scifit.tests.test_solvers_linear import GenericLinearRegression
