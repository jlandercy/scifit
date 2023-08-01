from scifit.interfaces.generic import FitSolverInterface


class LinearFitSolver(FitSolverInterface):

    @staticmethod
    def model(x, a, b):
        return a*x + b
