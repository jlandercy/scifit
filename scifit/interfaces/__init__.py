"""
Module :py:mod:`scifit.interfaces` defines all interfaces for the package.
Interface defines a standard and unique way to solve a specific problem and exposes conveniences
to interact and render the solution.

Solving a specific problem is then as simple as choosing the right interface to subclass
and implement the missing model to solve.

Benefits are important, it enforces consistency among models and ensure quality is standard while
it allows ease of reuse for next problems it also helps to create clean code.
"""

from scifit.interfaces.generic import *
