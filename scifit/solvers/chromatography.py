import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pybaselines import Baseline, utils
from scipy import signal, integrate


class ChromatogramSolver:

    def __init__(self, poly_order=3, prominence=1.0, width=1.0, distance=1.0):
        self._poly_order = poly_order
        self._prominence = prominence
        self._width = width
        self._distance = distance

    @staticmethod
    def random_peaks(
        n_peaks=10,
        xmin=0.0,
        xmax=1000.0,
        sx=20.0,
        hmin=5.0,
        hmax=15.0,
        wmin=5.0,
        wmax=15.0,
        seed=None,
    ):
        """
        Generate random peak coordinates:

        :param n_peaks:
        :param xmin:
        :param xmax:
        :param sx:
        :param hmin:
        :param hmax:
        :param wmin:
        :param wmax:
        :return:
        """
        np.random.seed(seed)
        peak_shifts = sx * np.random.normal(size=n_peaks)
        peaks = np.linspace(xmin + 10.0 * wmax, xmax - 10.0 * wmax, n_peaks)
        widths = np.random.uniform(low=wmin, high=wmax, size=n_peaks)
        heights = np.random.uniform(low=hmin, high=hmax, size=n_peaks)
        config = {
            "peaks": peaks + peak_shifts,
            "heights": heights,
            "widths": widths,
        }
        return config

    @staticmethod
    def synthetic_dataset(
        peaks=None,
        widths=None,
        heights=None,
        sy=0.10,
        n_peaks=10,
        xmin=0.0,
        xmax=1000.0,
        sx=20.0,
        hmin=5.0,
        hmax=15.0,
        wmin=5.0,
        wmax=15.0,
        resolution=5001,
        seed=None,
        mode="exp",
        b0=5.0,
        b1=15.0,
    ):
        """
        Create synthetic datasets
        Adapted from: https://github.com/derb12/pybaselines/blob/main/examples/spline/example_helpers.py

        :param resolution:
        :param mode:
        :return:
        """

        np.random.seed(seed)
        config = ChromatogramSolver.random_peaks(
            n_peaks=n_peaks,
            xmin=xmin,
            xmax=xmax,
            sx=sx,
            hmin=hmin,
            hmax=hmax,
            wmin=wmin,
            wmax=wmax,
            seed=seed,
        )

        if peaks is None:
            peaks = config["peaks"]
        peaks = np.array(peaks)

        if heights is None:
            heights = config["heights"]
        heights = np.array(heights)

        if widths is None:
            widths = config["widths"]
        widths = np.array(widths)

        x = np.linspace(xmin, xmax, resolution)

        y = np.full(x.shape, 0.0)
        for peak, height, width in zip(peaks, heights, widths):
            y += utils.gaussian(x, height, peak, width)

        if mode == "exp":
            baseline = b0 + b1 * np.exp(-x / (xmax - xmin))

        elif mode == "lin":
            baseline = b0 + b1 * x / (xmax - xmin)

        else:
            raise ValueError(f"Unknown baseline mode {mode}")

        noise = sy * np.random.normal(size=resolution)

        data = pd.DataFrame(
            {
                "x0": x,
                "y": y + baseline + noise,
                "yb": y + noise,
                "ybn": y,
                "b": baseline,
                "n": noise,
            }
        )

        return data

    @staticmethod
    def clean_base_indices(lefts, rights):
        """
        Clean buggy peak base limits
        :param lefts:
        :param rights:
        :return:
        """
        lefts = np.copy(lefts)
        rights = np.copy(rights)
        for i in range(len(lefts)-1):
            if lefts[i] == lefts[i+1]:
                lefts[i+1] = rights[i]
            if rights[i] == rights[i+1]:
                rights[i] = lefts[i+1]
        return lefts, rights

    def integrate_peaks(self):
        integrals = []
        for left, right in zip(self._peaks["lefts"], self._peaks["rights"]):
            x = self._xdata[left:right]
            y = self._ydata[left:right]
            integral = integrate.trapezoid(y, x)
            integrals.append(integral)
        return np.array(integrals)

    def fit(self, xdata, ydata=None):
        if isinstance(xdata, pd.DataFrame):
            ydata = xdata["y"].values
            xdata = xdata["x0"].values

        # Withdraw baseline:
        fitter = Baseline(xdata, check_finite=True)
        background = fitter.modpoly(ydata, poly_order=self._poly_order)
        baseline = background[0]
        filtered = ydata - baseline

        # Detect peaks:
        peaks = signal.find_peaks(
            filtered, prominence=self._prominence, width=self._width, distance=self._distance
        )
        meta = peaks[1]
        meta["indices"] = peaks[0]
        meta["times"] = xdata[peaks[0]]
        meta["lefts"], meta["rights"] = self.clean_base_indices(meta["left_bases"], meta["right_bases"])

        self._xdata = xdata
        self._ydata = ydata
        self._baseline = baseline
        self._filtered = filtered
        self._peaks = meta

        self._peaks["surfaces"] = self.integrate_peaks()

        return {"x0": xdata, "y": ydata, "b": baseline, "yb": filtered, "peaks": meta}

    def plot_fit(self):
        fig, axe = plt.subplots()

        axe.plot(self._xdata, self._ydata, label="Data")
        axe.plot(self._xdata, self._baseline, label="Baseline")

        axe.plot(
            self._xdata[self._peaks["indices"]],
            self._ydata[self._peaks["indices"]],
            linestyle="none",
            marker=".",
            label="Peaks",
        )
        axe.plot(
            self._xdata[self._peaks["lefts"]],
            self._ydata[self._peaks["lefts"]],
            linestyle="none",
            marker="x",
            label="Left Bases",
        )
        axe.plot(
            self._xdata[self._peaks["rights"]],
            self._ydata[self._peaks["rights"]],
            linestyle="none",
            marker="+",
            label="Right Bases",
        )

        for left, right in zip(self._peaks["lefts"], self._peaks["rights"]):
            axe.fill_between(
                self._xdata[left:right], self._ydata[left:right], self._baseline[left:right],
                color="blue", alpha=0.25
            )

        axe.set_title("Chromatogram Fit:")
        axe.set_xlabel(r"Time, $t$")
        axe.set_ylabel(r"Signal, $g(t)$")

        axe.legend()
        axe.grid()

        return axe
