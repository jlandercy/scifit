import os
import itertools
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scifit.errors.base import *
from scifit.interfaces.generic import FitSolverInterface


# Tests setup:
print_fit = bool(int(os.getenv("TESTS_PRINT_FIT", 1)))
print_chi2 = bool(int(os.getenv("TESTS_PRINT_CHI2", 1)))
print_k2s = bool(int(os.getenv("TESTS_PRINT_K2S", 1)))
print_loss_contour = bool(int(os.getenv("TESTS_PRINT_LOSS_CONTOUR", 0)))
print_loss_surface = bool(int(os.getenv("TESTS_PRINT_LOSS_SURFACE", 0)))
print_loss_iterations = bool(int(os.getenv("TESTS_PRINT_LOSS_ITERATIONS", 0)))


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
    data_path = None

    factory = None
    configuration = {}
    parameters = np.array([2.0, 3.0])

    xmin = -1.0
    xmax = +1.0
    mode = "lin"
    resolution = 30
    dimension = 1
    xdata = None

    seed = 789
    sigma = None
    scale_mode = "auto"
    generator = np.random.normal
    target_kwargs = {}
    sigma_factor = 10.0

    log_x = False
    log_y = False

    loss_domains = None
    loss_ratio = 10.0
    loss_factor = 3.0
    loss_log_x = False
    loss_log_y = False
    log_loss = False
    loss_resolution = 75

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

        if self.data_path is None:
            data = self.solver.synthetic_dataset(
                xdata=self.xdata,
                parameters=self.parameters,
                xmin=self.xmin,
                xmax=self.xmax,
                dimension=self.dimension,
                resolution=self.resolution,
                sigma=self.sigma,
                scale_mode=self.scale_mode,
                generator=self.generator,
                seed=self.seed,
                **self.target_kwargs,
            )

            self.xdata = data.filter(regex="^x").values
            self.ydata = data["y"].values
            self.sigmas = data["sy"].values
            self.yref = data["yref"].values
            self.ynoise = data["ynoise"].values

        else:
            data = self.solver.load(self.data_path, store=False)
            self.xdata = data.filter(regex="^x").values
            self.ydata = data["y"]

            if self.sigma is None and "sy" in data.columns:
                self.sigmas = data["sy"]

            else:
                self.sigmas = self.solver.generate_noise(
                    self.ydata,
                    sigma=self.sigma,
                    scale_mode=self.scale_mode,
                    generator=self.generator,
                    seed=self.seed,
                    full_output=True,
                    **self.target_kwargs,
                )["sigmas"]

        # Problem: Minimize & CurveFit does not beahve the same and cruvefit might also guess p0 or bar1 is better
        # Tu essayes de faire deux choses en même temps:
        # - faire que minimize & curve_fit se ressemble pour avoir l'accès au callback
        # - créer un p0 qui est bien pour l'affichage en mélangeant avec p0 qui sert au deux minimize et curve_fit

        # print(self.parameters)
        # print(self.loss_domains)
        # if self.parameters is not None:
        #     if self.loss_domains is None:
        #         domains = self.solver.parameter_domains(parameters=self.parameters)
        #     else:
        #         domains = self.loss_domains
        #     #self.configuration["p0"] = 0.5 * np.array(self.parameters)
        #     self.configuration["p0"] = domains.loc["min", :] + 0.25 * (domains.loc["max", :] - domains.loc["min", :])
        # print(self.configuration)

    def test_signature(self):
        s = self.solver.signature
        n = self.solver.parameter_space_size
        self.assertEqual(len(s.parameters) - 1, n)

    def test_model_implementation(self):
        if self.parameters is not None:
            yhat = self.solver.model(self.xdata, *self.parameters)
            self.assertTrue(np.allclose(yhat, self.yref))
            self.assertTrue(np.allclose(yhat + self.ynoise, self.ydata))

    def test_model_fit_stored_fields(self):
        solution = self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        for key in self.solver._data_keys + self.solver._result_keys:
            self.assertTrue(hasattr(self.solver, key))

    def test_clean_stored_fields(self):
        solution = self.solver.store(self.xdata, self.ydata, sigma=self.sigmas)
        self.solver.clean_results()
        for key in self.solver._result_keys:
            self.assertFalse(hasattr(self.solver, key))

    def test_parameter_domains_size(self):
        domains = self.solver.parameter_domains(
            parameters=self.parameters, mode=self.mode
        )
        self.assertEqual(self.solver.k, domains.shape[1])
        self.assertEqual(self.solver.k, domains.loc["min", :].size)
        self.assertEqual(self.solver.k, domains.loc["max", :].size)

    def test_model_solve_signature_no_sigma(self):
        solution = self.solver.solve(self.xdata, self.ydata, sigma=None)
        self.assertIsInstance(solution, dict)
        self.assertSetEqual(
            {"success", "parameters", "covariance", "info", "message", "status"},
            set(solution.keys()),
        )

    def test_model_solve_signature_sigma(self):
        solution = self.solver.solve(self.xdata, self.ydata, sigma=self.sigma)
        self.assertIsInstance(solution, dict)
        self.assertSetEqual(
            {"success", "parameters", "covariance", "info", "message", "status"},
            set(solution.keys()),
        )

    def test_model_solve_signature_sigmas(self):
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
        if self.parameters is not None:
            solution = self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
            for i in range(self.parameters.shape[0]):
                self.assertTrue(
                    np.allclose(
                        self.parameters[i],
                        solution["parameters"][i],
                        atol=self.sigma_factor * np.sqrt(solution["covariance"][i][i]),
                    )
                )

    def _test_model_minimize_against_solve(self):
        if self.parameters is not None:
            np.random.seed(self.seed)
            solution = self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)

            np.random.seed(self.seed)
            minimized = self.solver.minimize(self.xdata, self.ydata, sigma=None)

            # Assert both solve and minimize are alike at percent level
            for i in range(self.parameters.shape[0]):
                self.assertTrue(
                    np.allclose(
                        solution["parameters"][i],
                        minimized["parameters"][i],
                        rtol=5e-3,
                    )
                )

            # Assert covariance
            # for i in range(self.parameters.shape[0]):
            #     self.assertTrue(
            #         np.allclose(
            #             solution["covariance"][i][i],
            #             minimized["covariance"][i][i],
            #             rtol=5e-3,
            #         )
            #     )

    def test_goodness_of_fit(self):
        """
        Perform Chi 2 Test for Goodness of fit and check proper fits get their pvalue acceptable
        """
        solution = self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        test = self.solver.goodness_of_fit()
        self.assertIsInstance(test, dict)
        self.assertEqual(
            set(test.keys()).intersection({"statistic", "pvalue"}),
            {"statistic", "pvalue"},
        )
        if self.sigma is not None and self.sigma > 0.0:
            self.assertTrue(0.50 <= test["normalized"] <= 1.50)
            self.assertTrue(test["pvalue"] >= 0.10)

    def test_kolmogorov(self):
        solution = self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        test = self.solver.kolmogorov()
        self.assertIsInstance(test, dict)
        self.assertEqual(
            set(test.keys()).intersection({"statistic", "pvalue"}),
            {"statistic", "pvalue"},
        )
        if self.sigma is not None and self.sigma > 0.0:
            self.assertTrue(test["pvalue"] >= 0.10)

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

    def test_parameters_domain_from_iterations(self):
        solution = self.solver.fit(self.xdata, self.ydata)
        domains = self.solver.parameter_domains(iterations=True)

    def test_load_and_store(self):
        if self.data_path:
            data = self.solver.load(self.data_path, store=True)
            self.assertTrue(self.solver.stored(error=False))
            self.assertEqual(data.shape[0], self.solver._xdata.shape[0])
            self.assertEqual(data.shape[0], self.solver._ydata.shape[0])

    def test_dataset(self):
        self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        data = self.solver.dataset()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.index.name, "id")
        x = data.filter(regex="x")
        self.assertTrue(x.shape[1] > 0)
        keys = {"y", "sy", "yhat", "yerr", "yerrrel", "yerrabs", "yerrsqr", "chi2"}
        self.assertEqual(set(data.columns).intersection(keys), keys)

    def test_synthetic_dataset(self):
        data = self.solver.synthetic_dataset(
            parameters=self.parameters, dimension=self.dimension, sigma=self.sigma
        )
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.index.name, "id")
        x = data.filter(regex="x")
        self.assertTrue(x.shape[1] > 0)
        keys = {"y"}
        self.assertEqual(set(data.columns).intersection(keys), keys)

    def test_fit_from_synthetic_dataset(self):
        if self.parameters is None:
            parameters = np.random.uniform(size=(self.solver.k,), low=0.1, high=0.9)
        else:
            parameters = self.parameters
        data = self.solver.synthetic_dataset(
            xdata=self.xdata,
            parameters=parameters,
            dimension=self.dimension,
            sigma=self.sigma,
        )
        data = data.dropna(how="all", axis=1)
        self.solver.store(data=data)
        solution = self.solver.fit(p0=0.75 * parameters)
        self.assertTrue(
            np.allclose(
                parameters,
                solution["parameters"],
                atol=1e-6,
                rtol=10. * (self.sigma or 1e-4),
            )
        )

    def test_fitted_dataset(self):
        self.solver.store(self.xdata, self.ydata, sigma=self.sigmas)
        data = self.solver.dataset()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.index.name, "id")
        x = data.filter(regex="x")
        self.assertTrue(x.shape[1] > 0)
        keys = {"y"}
        self.assertEqual(set(data.columns).intersection(keys), keys)

    def test_dataset_serialization_equivalence(self):
        name = self.__class__.__name__
        file1 = "{}/{}.csv".format(self.media_path, name)
        # file2 = "{}/{}_echo.csv".format(self.media_path, name)

        np.random.seed(self.seed)
        solution1 = self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        check1 = self.solver.dataset()
        self.solver.dump(file1, summary=False)

        solver2 = self.factory(**self.configuration)
        solver2.load(file1, store=True)
        np.random.seed(self.seed)
        solution2 = solver2.fit()
        check2 = solver2.dataset()
        # solver2.dump(file2, summary=False)

        # Assert 3 significant digits
        self.assertTrue(
            np.allclose(
                solution1["parameters"], solution2["parameters"], atol=1e-6, rtol=1e-3
            )
        )
        self.assertTrue(
            np.allclose(
                check1.values, check2.values, equal_nan=True, atol=1e-6, rtol=1e-3
            )
        )

    def test_random_seed_synthetic_dataset_reproducibility(self):
        seed = np.random.randint(low=1, high=999999999, size=1)[0]
        if self.parameters is None:
            parameters = np.random.uniform(size=(self.solver.k,), low=0.1, high=0.9)
        else:
            parameters = self.parameters
        dataset1 = self.solver.synthetic_dataset(
            dimension=self.dimension or self.xdata.shape[1],
            parameters=parameters,
            resolution=250,
            sigma=self.sigma or 0.015,
            seed=seed,
        ).dropna(how="all", axis=1)
        dataset2 = self.solver.synthetic_dataset(
            dimension=self.dimension or self.xdata.shape[1],
            parameters=parameters,
            resolution=250,
            sigma=self.sigma or 0.015,
            seed=seed,
        ).dropna(how="all", axis=1)
        self.assertTrue(np.allclose(dataset1.values, dataset2.values, equal_nan=True))
        self.assertTrue(dataset1.equals(dataset2))

    def test_dump_summary(self):
        name = self.__class__.__name__
        self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
        self.solver.dump(
            "{}/{}_summary.csv".format(self.media_path, name), summary=True
        )

    def test_plot_fit(self):
        if print_fit:
            name = self.__class__.__name__
            title = r"{} (seed={:d})".format(name, self.seed)
            self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
            axe = self.solver.plot_fit(
                title=title,
                errors=True,
                squared_errors=False,
                mode=self.mode,
                log_x=self.log_x,
                log_y=self.log_y,
            )
            axe.figure.savefig("{}/{}_fit.{}".format(self.media_path, name, self.format))
            plt.close(axe.figure)

    def test_plot_chi_square(self):
        if print_chi2:
            name = self.__class__.__name__
            title = r"{} (seed={:d})".format(name, self.seed)
            self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
            axe = self.solver.plot_chi_square(title=title)
            axe.figure.savefig("{}/{}_chi2.{}".format(self.media_path, name, self.format))
            plt.close(axe.figure)

    def test_plot_kolmogorov(self):
        if print_k2s:
            name = self.__class__.__name__
            title = r"{} (seed={:d})".format(name, self.seed)
            self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
            axe = self.solver.plot_kolmogorov(title=title)
            axe.figure.savefig("{}/{}_k2s.{}".format(self.media_path, name, self.format))
            plt.close(axe.figure)

    def test_plot_loss_automatic(self):
        if print_loss_contour:
            name = self.__class__.__name__
            title = r"{} (seed={:d})".format(name, self.seed)
            self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)
            if self.loss_domains is None:
                domains = self.solver.parameter_domains(iterations=print_loss_iterations)
            else:
                domains = self.loss_domains
            axe = self.solver.plot_loss(
                title=title,
                iterations=print_loss_iterations,
                mode=self.mode,
                domains=domains,
                ratio=self.loss_ratio,
                factor=self.loss_factor,
                log_x=self.loss_log_x,
                log_y=self.loss_log_y,
                log_loss=self.log_loss,
            )
            if self.solver.k > 2:
                axe = axe[0][0]
            axe.figure.savefig(
                "{}/{}_loss_scatter.{}".format(self.media_path, name, self.format)
            )
            plt.close(axe.figure)

    def test_plot_loss_surface_automatic(self):
        if print_loss_surface:
            name = self.__class__.__name__
            title = r"{} (seed={:d})".format(name, self.seed)
            self.solver.fit(self.xdata, self.ydata, sigma=self.sigmas)

            if self.solver.k <= 2:

                axe = self.solver.plot_loss_low_dimension(
                    title=title,
                    surface=True,
                    add_title=True,
                    iterations=False,
                    resolution=self.loss_resolution,
                    domains=self.loss_domains,
                    ratio=self.loss_ratio,
                    factor=self.loss_factor,
                    log_x=False,
                    log_y=False,
                    log_loss=False,
                )
                axe.figure.savefig(
                    "{}/{}_loss_surface_b1_b2.{}".format(
                        self.media_path, name, self.format
                    )
                )
                plt.close(axe.figure)

            else:

                for axe in self.solver.plot_loss(
                    title=title,
                    surface=True,
                    iterations=False,
                    resolution=self.loss_resolution,
                    domains=self.loss_domains,
                    ratio=self.loss_ratio,
                    factor=self.loss_factor,
                    log_x=False,
                    log_y=False,
                    log_loss=False,
                ):
                    i, j = axe._pair_indices
                    axe.figure.savefig(
                        "{}/{}_loss_surface_b{}_b{}.{}".format(
                            self.media_path, name, i + 1, j + 1, self.format
                        )
                    )
                    plt.close(axe.figure)
