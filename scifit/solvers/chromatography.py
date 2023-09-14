import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pybaselines import Baseline, utils
from scipy import integrate, signal

from scifit.toolbox.report import ChromatogramSolverReportProcessor


class ChromatogramSolver:
    def __init__(
        self,
        mode="imodpoly",
        configuration=None,
        prominence=1.0,
        width=10.0,
        height=None,
        distance=None,
    ):
        # Baseline Configuration:
        self._mode = mode
        self._configuration = configuration or {}

        # Peak Detection:
        self._prominence = prominence
        self._height = height
        self._width = width
        self._distance = distance

        # Peak Integration:

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
        Clean buggy peak base limits to make them strictly monotonically increasing
        :param lefts:
        :param rights:
        :return:
        """
        _lefts = np.copy(lefts)
        _rights = np.copy(rights)
        for i in range(len(lefts) - 1):
            if lefts[i + 1] < lefts[i]:
                _lefts[i + 1] = rights[i]
            if rights[i + 1] < rights[i]:
                _rights[i] = lefts[i + 1]
            if lefts[i] == lefts[i + 1]:
                _lefts[i + 1] = rights[i]
            if rights[i] == rights[i + 1]:
                _rights[i] = lefts[i + 1]
        return _lefts, _rights

    def integrate_peaks(self):
        """
        Integrate each peak
        :return:
        """
        integrals = []
        for left, right in zip(self._peaks["lefts"], self._peaks["rights"]):
            x = self._xdata[left:right]
            y = self._filtered[left:right]
            integral = integrate.trapezoid(y, x)
            integrals.append(integral)
        return np.array(integrals)

    def estimate_noise(self):
        """
        Statistically estimate baseline noise outside peaks bases
        :return:
        """
        y = np.copy(self._filtered)
        for left, right in zip(self._peaks["lefts"], self._peaks["rights"]):
            y[left:right] = np.nan
        valid = ~np.isnan(y)
        data = {
            "selected": np.sum(valid.astype(int)),
            "size": y.size,
            "mean": y[valid].mean(),
            "std": y[valid].std(),
        }
        data["ratio"] = data["selected"] / data["size"]
        data["LOD"] = data["mean"] + 3.0 * data["std"]
        data["LOQ"] = data["mean"] + 10.0 * data["std"]
        return data

    def solve(self, xdata, ydata, rel_height=0.5):
        # Solve peaks:
        peaks = signal.find_peaks(
            ydata,
            prominence=self._prominence,
            width=self._width,
            distance=self._distance,
            height=self._height,
            rel_height=rel_height,
        )
        meta = peaks[1]
        meta["indices"] = peaks[0]
        meta["times"] = xdata[peaks[0]]

        # Clean buggy base indices:
        meta["lefts"], meta["rights"] = self.clean_base_indices(
            meta["left_bases"], meta["right_bases"]
        )

        # Rescale widths:
        factor = (xdata.max() - xdata.min()) / xdata.size
        meta["left_ips"] *= factor
        meta["right_ips"] *= factor
        meta["widths"] = meta["right_ips"] - meta["left_ips"]

        return meta

    def fit(self, xdata, ydata=None, dead_time=0.0):
        """
        Fit chromatogram and describe peaks:

          - Remove baseline
          - Find peaks retention times and boundaries
          - Integrate peaks over boundaries
          - Assess baseline noise, LOD and LOQ

        :param xdata:
        :param ydata:
        :return:
        """
        if isinstance(xdata, pd.DataFrame):
            ydata = xdata["y"].values
            xdata = xdata["x0"].values

        # Select baseline filter:
        fitter = Baseline(xdata, check_finite=True)
        bfilter = getattr(fitter, self._mode)

        # Configure baseline filter:
        background = bfilter(ydata, **self._configuration)
        baseline = background[0]

        # Withdraw baseline
        filtered = ydata - baseline

        # Detect peaks:
        meta = self.solve(xdata=xdata, ydata=filtered, rel_height=0.5)

        # Compute Chromatography Quantities:

        # Plateau Numbers
        meta["N"] = 5.54 * np.power(meta["times"] / meta["widths"], 2)

        # Asymmetry:
        meta_10H = self.solve(xdata=xdata, ydata=filtered, rel_height=0.1)
        meta["a10"] = meta_10H["times"] - meta_10H["left_ips"]
        meta["b10"] = meta_10H["right_ips"] - meta_10H["times"]
        meta["As"] = meta["b10"] / meta["a10"]

        # Tailing:
        meta_20H = self.solve(xdata=xdata, ydata=filtered, rel_height=0.05)
        meta["a20"] = meta_20H["times"] - meta_20H["left_ips"]
        meta["T"] = meta_20H["widths"] / (2.0 * meta["a20"])

        # Resolution:
        meta["R"] = (
            2.0
            * (meta["times"] - meta["times"][0])
            / (meta_20H["widths"] + meta_20H["widths"][0])
        )

        # Selectivity:
        meta["alpha"] = (meta["times"] - dead_time) / (meta["times"][0] - dead_time)

        # Store:
        self._xdata = xdata
        self._ydata = ydata
        self._baseline = baseline
        self._filtered = filtered
        self._peaks = meta

        # Integrate:
        self._peaks["surfaces"] = self.integrate_peaks()

        # Estimate baseline noise:
        self._noise = self.estimate_noise()

        return {"x0": xdata, "y": ydata, "b": baseline, "yb": filtered, "peaks": meta}

    def summary(self):
        data = pd.DataFrame(self._peaks)
        data = data.reindex(
            [
                "times",
                "prominences",
                "widths",
                "surfaces",
                "N",
                "R",
                "alpha",
                "As",
                "T",
            ],
            axis=1,
        )
        return data

    def plot_fit(self, title="", surfaces=True, heights=False, widths=False):
        fig, axe = plt.subplots()

        axe.plot(self._xdata, self._ydata, label="Data")
        axe.plot(self._xdata, self._baseline, label="Baseline")
        axe.plot(
            self._xdata,
            self._baseline + self._noise["LOQ"],
            "--",
            label="LOQ = {:.2f}".format(self._noise["LOQ"]),
        )
        axe.plot(
            self._xdata,
            self._baseline + self._noise["LOD"],
            "--",
            label="LOD = {:.2f}".format(self._noise["LOD"]),
        )

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

        for peak, time, surface, left, right in zip(
            self._peaks["indices"],
            self._peaks["times"],
            self._peaks["surfaces"],
            self._peaks["lefts"],
            self._peaks["rights"],
        ):
            axe.text(
                self._xdata[peak],
                self._ydata[peak],
                "{:.1f}".format(time),
                fontsize=7,
                horizontalalignment="center",
            )

            if surfaces:
                axe.fill_between(
                    self._xdata[left:right],
                    self._ydata[left:right],
                    self._baseline[left:right],
                    color="blue",
                    alpha=0.25,
                )
                axe.text(
                    self._xdata[peak],
                    (self._baseline[peak] + 1.5 * self._noise["LOQ"]),
                    "{:.1f}".format(surface),
                    fontsize=7,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    rotation=90,
                )

        if widths:
            axe.vlines(
                x=self._xdata[self._peaks["indices"]],
                ymin=self._ydata[self._peaks["indices"]] - self._peaks["prominences"],
                ymax=self._ydata[self._peaks["indices"]],
                color="yellow",
            )

        if heights:
            axe.hlines(
                y=self._baseline[self._peaks["indices"]] + self._peaks["width_heights"],
                xmin=self._peaks["left_ips"],
                xmax=self._peaks["right_ips"],
                color="yellow",
            )

        axe.set_title("Chromatogram Fit:\n%s" % title)
        axe.set_xlabel(r"Time, $t$")
        axe.set_ylabel(r"Signal, $g(t)$")

        axe.legend(bbox_to_anchor=(1, 1), loc="upper left")
        axe.grid()

        fig.subplots_adjust(right=0.75)

        return axe

    def report(self, file, path=".", mode="pdf", **kwargs):
        processor = ChromatogramSolverReportProcessor()
        processor.report(self, file=file, path=path, mode=mode, **kwargs)
