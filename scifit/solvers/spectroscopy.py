import inspect
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pybaselines import Baseline, utils
from scipy import optimize, stats, signal

from scifit.solvers.chromatography import ChromatogramSolver


class SpectroscopySolver:

    @staticmethod
    def peak_model(x, loc, scale, height):
        law = stats.cauchy(loc=loc, scale=scale)
        return height * law.pdf(x) / law.pdf(loc)

    @classmethod
    def m(cls):
        return len(inspect.signature(cls.peak_model).parameters) - 1

    @classmethod
    def model(cls, x, *p):
        m = cls.m()
        n = len(p) // m
        y = np.zeros_like(x)
        for i in range(n):
            y += cls.peak_model(x, *p[i * m:(i + 1) * m])
        return y

    @classmethod
    def synthetic_dataset(cls, sigma=0.005):

        x0s = [10, 25, 70, 100, 120, 140]
        s0s = [1, 1.5, 2, 1, 0.5, 1]
        A0s = [2, 3, 2, 3, 1, 1]
        p0 = list(itertools.chain(*zip(x0s, s0s, A0s)))
        # [10, 1, 2, 25, 1.5, 3, 70, 2, 2, 100, 1, 3, 120, 0.5, 1, 140, 1, 1]

        np.random.seed(123456)
        x = np.linspace(0, 150, 500)
        y = cls.model(x, *p0)
        sy = sigma * np.ones_like(x)
        noise = sy * np.random.normal(size=sy.size)

        data = pd.DataFrame(
            {
                "x0": x,
                "y": y + noise,
                "sy": sy,
                "n": noise,
            }
        )

        return data

    @classmethod
    def loss_factory(cls, xdata, ydata, sigma=None):

        if sigma is None:
            sigma = 1.

        def wrapped(p):
            return np.sum(np.power((ydata - cls.model(xdata, *p)) / sigma, 2)) / (xdata.size - len(p))

        return wrapped

    def solve(self, xdata, ydata, sigma=None, baseline_mode="loess", prominence=1., distance=1., width=1.):

        baseline_fitter = getattr(Baseline(x_data=xdata), baseline_mode)

        baseline, params = baseline_fitter(ydata)
        yb = ydata - baseline

        peaks, bases = signal.find_peaks(yb, prominence=prominence, distance=distance, width=width)
        #lefts, rights = ChromatogramSolver.clean_base_indices(bases["left_bases"], bases["right_bases"])
        lefts, rights = bases["left_bases"], bases["right_bases"]
        print("Found %d peak(s): %s" % (len(peaks), peaks))

        x0 = list(itertools.chain(*[[xdata[i], w / 2., p] for i, w, p in zip(peaks, bases["widths"], bases["prominences"])]))
        bounds = list(itertools.chain(*[
            [(xdata[lefts[i]], xdata[rights[i]])] + [(0., np.inf)] * (self.m() - 1) for i in range(len(peaks))
        ]))

        loss = self.loss_factory(xdata, yb, sigma)
        solution = optimize.minimize(loss, x0=x0, bounds=bounds)

        print(x0)
        print(solution.x)

        return {
            "indices": peaks,
            "coordinates": xdata[peaks],
            "lefts": lefts,
            "rights": rights,
            "parameters": {
                "x0": x0,
                "bounds": bounds,
            },
            "solution": solution,
            "baseline": baseline,
            "yhat": self.model(xdata, *solution.x),
        }

    def fit(self, xdata, ydata=None, sigma=None, prominence=1., distance=1., width=1.):

        if isinstance(xdata, pd.DataFrame):
            ydata = xdata["y"].values
            if "sy" in xdata:
                sigma = xdata["sy"].values
            xdata = xdata["x0"].values

        solution = self.solve(xdata, ydata, sigma, prominence=prominence, distance=distance, width=width)

        self._xdata = xdata
        self._ydata = ydata
        self._sigma = sigma
        self._solution = solution
        self._yhat = solution["yhat"]

        return solution

    def plot_fit(self, title=None):
        fig, axe = plt.subplots()
        axe.plot(self._xdata, self._ydata)
        axe.plot(self._xdata, self._yhat)
        axe.grid()
        return axe
