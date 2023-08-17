import itertools
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scifit.errors.base import *
from scifit.interfaces.generic import FitSolverInterface


class GenericTestFitSolverInterface:
    xdata = None
    ydata = None

    def setUp(self) -> None:
        self.solver = FitSolverInterface()
        self.solver.store(self.xdata, self.ydata)

    def test_missing_model(self):
        with self.assertRaises(MissingModel):
            self.solver.model([], [])

    def test_space_sizes(self):
        self.assertEqual(self.solver.observation_space_size, self.solver.n)
        self.assertEqual(self.solver.feature_space_size, self.solver.m)
        self.assertEqual(self.solver.parameter_space_size, self.solver.k)
        self.assertEqual(self.solver.n, self.solver._xdata.shape[0])
        self.assertEqual(self.solver.m, self.solver._xdata.shape[1])
        self.assertEqual(self.solver.k, len(self.solver.signature.parameters) - 1)

    def test_feature_domains(self):
        data = self.solver.feature_domains()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape[1], self.solver.m)
        self.assertEqual(set(data.index).intersection({"min", "max"}), {"min", "max"})

    def test_linear_space_generation(self):
        xlin = self.solver.scale(mode="lin", xmin=-10.0, xmax=+10.0, resolution=200)
        self.assertIsInstance(xlin, np.ndarray)
        self.assertEqual(xlin.ndim, 1)
        self.assertEqual(xlin.shape[0], 200)
        self.assertTrue(np.allclose(xlin.min(), -10.0))
        self.assertTrue(np.allclose(xlin.max(), +10.0))

    def test_logarithmic_space_generation(self):
        xlog = self.solver.scale(mode="log", xmin=1e-10, xmax=1e10, resolution=200)
        self.assertIsInstance(xlog, np.ndarray)
        self.assertEqual(xlog.ndim, 1)
        self.assertEqual(xlog.shape[0], 200)
        self.assertTrue(np.allclose(xlog.min(), 1e-10))
        self.assertTrue(np.allclose(xlog.max(), 1e10))

    def test_feature_scales(self):
        scales = self.solver.feature_scales(resolution=200)
        self.assertIsInstance(scales, list)
        self.assertEqual(len(scales), self.solver.m)
        for i in range(self.solver.m):
            self.assertIsInstance(scales[i], np.ndarray)
            self.assertEqual(scales[i].shape[0], 200)

    def test_feature_space(self):
        spaces = self.solver.feature_space(resolution=10)
        self.assertIsInstance(spaces, list)
        self.assertEqual(len(spaces), self.solver.m)
        for i in range(self.solver.m):
            self.assertIsInstance(spaces[i], np.ndarray)
            self.assertEqual(spaces[i].ndim, self.solver.m)
            for k in range(self.solver.m):
                self.assertEqual(spaces[i].shape[k], 10)

    def test_feature_dataset_1D(self):
        dataset = self.solver.feature_dataset(dimension=1, resolution=10)
        self.assertIsInstance(dataset, np.ndarray)
        self.assertEqual(dataset.ndim, 2)
        self.assertEqual(dataset.shape[0], 10**1)
        self.assertEqual(dataset.shape[1], 1)

    def test_feature_dataset_2D(self):
        dataset = self.solver.feature_dataset(dimension=5, resolution=10)
        self.assertIsInstance(dataset, np.ndarray)
        self.assertEqual(dataset.ndim, 2)
        self.assertEqual(dataset.shape[0], 10**5)
        self.assertEqual(dataset.shape[1], 5)

    def test_feature_dataset_auto(self):
        dataset = self.solver.feature_dataset(
            domains=self.solver.feature_domains(), resolution=10
        )
        self.assertIsInstance(dataset, np.ndarray)
        self.assertEqual(dataset.ndim, 2)
        self.assertEqual(dataset.shape[0], 10**self.solver.m)
        self.assertEqual(dataset.shape[1], self.solver.m)

    def test_parameters_domains_auto_not_fitted(self):
        data = self.solver.parameter_domains()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape[1], self.solver.k)
        self.assertEqual(set(data.index).intersection({"min", "max"}), {"min", "max"})

    def test_parameters_domains_simple_fixed_not_fitted(self):
        data = self.solver.parameter_domains(xmin=-1.0, xmax=2)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape[1], self.solver.k)
        self.assertEqual(set(data.index).intersection({"min", "max"}), {"min", "max"})

    def test_parameters_domains_list_fixed_not_fitted(self):
        data = self.solver.parameter_domains(
            xmin=list(range(self.solver.k)), xmax=self.solver.k
        )
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape[1], self.solver.k)
        self.assertEqual(set(data.index).intersection({"min", "max"}), {"min", "max"})


class GenericTestFitSolver:
    root_path = ".cache/media/tests/"

    factory = None
    configuration = {}
    parameters = np.array([2.0, 3.0])

    xmin = -1.0
    xmax = +1.0
    mode = "lin"
    resolution = 30
    dimension = 1
    xdata = None
    p0 = None

    seed = 789
    sigma = None
    scale_mode = "auto"
    generator = np.random.normal
    target_kwargs = {}
    sigma_factor = 10.0

    format = "png"

    # scale_mode = "auto"
    # generator = np.random.uniform
    # target_kwargs = {"low": -.5, "high": .5}
    # sigma_factor = 10.

    def setUp(self) -> None:
        self.media_path = pathlib.Path(self.root_path) / format(
            self.factory.__module__.split(".")[-1]
        )
        self.media_path.mkdir(parents=True, exist_ok=True)

        self.solver = self.factory(**self.configuration)

        if self.xdata is None:
            self.xdata = self.solver.feature_dataset(
                mode=self.mode,
                xmin=self.xmin,
                xmax=self.xmax,
                dimension=self.dimension,
                resolution=self.resolution,
            )

        target = self.solver.target_dataset(
            self.xdata,
            *self.parameters,
            sigma=self.sigma,
            scale_mode=self.scale_mode,
            generator=self.generator,
            seed=self.seed,
            **self.target_kwargs,
            full_output=True,
        )

        self.__dict__.update(
            {k: target[k] for k in ["ydata", "yref", "sigmas", "ynoise"]}
        )

        domains = self.solver.parameter_domains(parameters=self.parameters)
        if self.p0 is None:
            self.p0 = domains.loc["max", :].values

    def test_signature(self):
        s = self.solver.signature
        n = self.solver.parameter_space_size
        self.assertEqual(len(s.parameters) - 1, n)

    def test_model_implementation(self):
        yhat = self.solver.model(self.xdata, *self.parameters)
        self.assertTrue(np.allclose(yhat, self.yref))
        self.assertTrue(np.allclose(yhat + self.ynoise, self.ydata))

    def test_model_fit_stored_fields(self):
        solution = self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        for key in ["_xdata", "_ydata", "_solution", "_yhat", "_score"]:
            self.assertTrue(hasattr(self.solver, key))

    def test_model_solve_signature(self):
        solution = self.solver.solve(self.xdata, self.ydata, sigma=self.sigmas)
        self.assertIsInstance(solution, dict)
        self.assertSetEqual(
            {"success", "parameters", "covariance", "info", "message", "status"},
            set(solution.keys()),
        )

    def test_model_minimize_signature(self):
        solution = self.solver.minimize(self.xdata, self.ydata, sigma=self.sigmas)
        self.assertIsInstance(solution, dict)
        self.assertSetEqual(
            {"success", "parameters", "covariance", "info", "message", "status"},
            set(solution.keys()),
        )

    def test_model_fit_parameters(self):
        """
        Check regressed parameters are equals up to `scale` Standard Deviation of fit precision
        Very unlikely to fail but tight enough to detect bad regression
         - Is the right solution
         - Is it precise enough wrt standard deviation
        """
        solution = self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        for i in range(self.parameters.shape[0]):
            self.assertTrue(
                np.allclose(
                    self.parameters[i],
                    solution["parameters"][i],
                    atol=self.sigma_factor * np.sqrt(solution["covariance"][i][i]),
                )
            )

    # def test_model_minimize_against_solve(self):
    #
    #     np.random.seed(self.seed)
    #     solution = self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
    #
    #     np.random.seed(self.seed)
    #     minimized = self.solver.minimize(self.xdata, self.ydata, sigma=None)
    #
    #     # Assert both solve and minimize are alike at percent level
    #     for i in range(self.parameters.shape[0]):
    #
    #         self.assertTrue(
    #             np.allclose(
    #                 solution["parameters"][i],
    #                 minimized["parameters"][i],
    #                 rtol=5e-3,
    #             )
    #         )
    #
    #     # Assert covariance
    #     # for i in range(self.parameters.shape[0]):
    #     #     self.assertTrue(
    #     #         np.allclose(
    #     #             solution["covariance"][i][i],
    #     #             minimized["covariance"][i][i],
    #     #             rtol=5e-3,
    #     #         )
    #     #     )

    def test_goodness_of_fit(self):
        """
        Perform Chi 2 Test for Goodness of fit and check proper fits get their pvalue acceptable
        """
        solution = self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        test = self.solver.goodness_of_fit(self.xdata, self.ydata, sigma=self.sigmas)
        self.assertIsInstance(test, dict)
        self.assertEqual(
            set(test.keys()).intersection({"statistic", "pvalue"}),
            {"statistic", "pvalue"},
        )
        # Ensure proper fits get its test valid:
        if self.sigma is not None and self.sigma > 0.0:
            self.assertTrue(0.50 <= test["normalized"] <= 1.50)
            self.assertTrue(test["pvalue"] >= 0.10)
        else:
            self.assertTrue(0.85 <= test["normalized"] <= 1.15)
            self.assertTrue(test["pvalue"] >= 0.50)

    def test_feature_dataset_auto(self):
        self.solver.store(self.xdata, self.ydata)
        dataset = self.solver.feature_dataset(
            domains=self.solver.feature_domains(), resolution=10
        )
        self.assertIsInstance(dataset, np.ndarray)
        self.assertEqual(dataset.ndim, 2)
        self.assertEqual(dataset.shape[0], 10**self.solver.m)
        self.assertEqual(dataset.shape[1], self.solver.m)

    def test_parameters_domain_linear_auto(self):
        solution = self.solver.fit(self.xdata, self.ydata)
        domains = self.solver.parameter_domains()

    def test_parameters_domain_linear_fixed(self):
        solution = self.solver.fit(self.xdata, self.ydata)
        domains = self.solver.parameter_domains(xmax=1e2)

    def test_parameters_domain_logarithmic_auto(self):
        solution = self.solver.fit(self.xdata, self.ydata)
        domains = self.solver.parameter_domains(mode="log", xmin=1e-5)

    def test_parameters_domain_logarithmic_fixed(self):
        solution = self.solver.fit(self.xdata, self.ydata)
        domains = self.solver.parameter_domains(mode="log", xmin=1e-5, xmax=100.0)

    def test_plot_fit(self):
        name = self.__class__.__name__
        title = r"{} (seed={:d})".format(name, self.seed)
        self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        axe = self.solver.plot_fit(title=title, errors=True, squared_errors=False)
        axe.figure.savefig(
            "{}/{}_fit.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def _test_plot_chi_square(self):
        name = self.__class__.__name__
        title = r"{} (seed={:d})".format(name, self.seed)
        self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        axe = self.solver.plot_chi_square(title=title)
        axe.figure.savefig("{}/{}_chi2.{}".format(self.media_path, name, self.format))
        plt.close(axe.figure)

    def _test_plot_loss_automatic(self):
        name = self.__class__.__name__
        title = r"{} (seed={:d})".format(name, self.seed)
        self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        axe = self.solver.plot_loss(title=title, iterations=False)
        axe.figure.savefig(
            "{}/{}_loss_scatter.{}".format(self.media_path, name, self.format)
        )
        plt.close(axe.figure)

    def _test_plot_loss_surface_automatic(self):
        name = self.__class__.__name__
        title = r"{} (seed={:d})".format(name, self.seed)
        self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)

        for i, j in itertools.combinations(range(self.solver.k), 2):
            axe = self.solver.plot_loss_low_dimension(
                title=title, first_index=i, second_index=j, surface=True, add_title=False, iterations=False
            )
            axe.figure.savefig(
                "{}/{}_loss_surface_b{}_b{}.{}".format(self.media_path, name, i+1, j+1, self.format)
            )
            plt.close(axe.figure)

    def test_dataset(self):
        self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        data = self.solver.dataset()

    def test_summary(self):
        name = self.__class__.__name__
        self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        data = self.solver.summary()
        data.to_csv("{}/{}.csv".format(self.media_path, name), index=True, sep=";")
