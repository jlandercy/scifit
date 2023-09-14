import base64
import io
import pathlib
from collections.abc import Iterable

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypandoc

from scifit.errors.base import *


class ReportProcessor:
    """
    Render report using pandoc
    """

    template = None
    directory = None

    @classmethod
    def context(cls, solver, *args, **kwargs):
        return {}

    @classmethod
    def render(cls, context=None, template=None, directory=None):
        template = template or cls.template
        directory = directory or pathlib.Path(__file__).parent / "resources/reports/"
        loader = jinja2.FileSystemLoader(searchpath=str(directory))
        environment = jinja2.Environment(loader=loader)
        template = environment.get_template(template)
        payload = template.render(**(context or {}))
        return payload

    @staticmethod
    def convert(payload, file="report", path=".", mode="pdf"):
        filename = pathlib.Path(path) / file
        pypandoc.convert_text(
            payload,
            mode,
            format="md",
            outputfile="%s.%s" % (filename, mode),
            extra_args=["--pdf-engine=pdflatex", "--biblatex"],
        )

    @staticmethod
    def serialize_figure(axe):
        stream = io.BytesIO()
        axe.figure.savefig(stream, format="svg")
        payload = "data:image/svg+xml;base64,{}".format(
            base64.b64encode(stream.getvalue()).decode()
        )
        return payload

    @staticmethod
    def serialize_table(frame, mode="latex", index=False):
        stream = io.StringIO()
        if mode == "latex":
            frame.columns = frame.columns.map("{{{}}}".format)
            frame.to_latex(
                stream,
                index=index,
                header=True,
                longtable=True,
                column_format="S" * frame.shape[1],
            )
        elif mode == "md":
            frame.to_markdown(stream, index=index)
        elif mode == "html":
            frame.to_html(stream, index=index)
        return stream.getvalue()

    @classmethod
    def serialize(cls, item, table_mode="latex"):
        if isinstance(item, plt.Axes):
            return cls.serialize_figure(item)

        elif isinstance(item, pd.DataFrame):
            return cls.serialize_table(item, mode=table_mode)

    @classmethod
    def report(
        cls, solver, context=None, path=".", file="report", mode="pdf", **kwargs
    ):
        modes = {"pdf", "html", "docx"}
        if mode not in modes:
            raise ConfigurationError(
                "Mode must be in %s, got '%s' instead" % (modes, mode)
            )
        if mode == "pdf":
            table_mode = "latex"
        elif mode == "html":
            table_mode = "html"
        else:
            table_mode = "md"
        context = cls.context(solver, context=context, table_mode=table_mode, **kwargs)
        payload = cls.render(context)
        cls.convert(payload, path=path, file=file, mode=mode)


class FitSolverReportProcessor(ReportProcessor):
    template = "fit.md"

    @classmethod
    def context(cls, solver, context=None, table_mode="latex", figure_mode="svg"):
        if context is None:
            context = {
                "title": "Fit Report",
                "author": "SciFit automatic report",
                "supervisor": "Jean Landercy",
            }

        axe = solver.plot_fit(errors=True, resolution=solver.n * 10)
        fit = cls.serialize(axe)
        plt.close(axe.figure)

        axe = solver.plot_loss()
        if isinstance(axe, Iterable):
            axe = axe[0][0]
        loss = cls.serialize(axe)
        plt.close(axe.figure)

        axe = solver.plot_chi_square()
        chi2 = cls.serialize(axe)
        plt.close(axe.figure)

        axe = solver.plot_kolmogorov()
        k2s = cls.serialize(axe)
        plt.close(axe.figure)

        # Dataset
        data = solver.dataset()
        data = data.drop(["yerrrel", "yerrabs"], axis=1)
        # Too bad pandas fails now for formatting
        for key in data:
            data[key] = data[key].apply("{:.4g}".format)
        data = data.reset_index()
        data = data.reindex(["id", "x0", "y", "sy", "yhat", "yerr", "chi2"], axis=1)
        data = data.rename(
            columns={
                "id": r"id",
                "x0": r"$x_0$",
                "x1": r"$x_1$",
                "x2": r"$x_2$",
                "y": r"$y$",
                "pm": r"$\pm$",
                "sy": r"$\sigma_y$",
                "yhat": r"$\hat{y}$",
                "yerr": r"$e$",
                "yerrrel": r"$e\hat{y}$",
                "yerrabs": r"$|e|$",
                "yerrsqr": r"$e^2$",
                "chi2": r"$\chi^2$",
            }
        )
        table = cls.serialize(data, table_mode=table_mode)

        # Fit parameters:
        parameters = solver.parameters()
        for key in parameters:
            parameters[key] = parameters[key].apply("{:.4g}".format)
        parameters = parameters.reset_index()
        if table_mode == "latex":
            parameters["pm"] = r"{$\pm$}"
        else:
            parameters["pm"] = r"$\pm$"
        parameters = parameters.reindex(["index", "b", "pm", "sb"], axis=1)
        parameters = parameters.rename(
            columns={
                "index": r"$i$",
                "b": r"$\beta_i$",
                "pm": r"$\pm$",
                "sb": r"$\sigma_{\beta_i}$",
            }
        )
        parameters = cls.serialize(parameters, table_mode=table_mode)

        # Chi 2 Abacus
        chi2_abacus = solver.chi_square_table()
        chi2_abacus = (
            chi2_abacus.groupby(["alpha", "key"])[["low-value", "high-value"]]
            .first()
            .unstack()
            .dropna(how="all", axis=1)
        )
        chi2_abacus = chi2_abacus.droplevel(0, axis=1)
        chi2_abacus.columns = [
            "Left one-sided",
            "Left two-sided",
            "Right one-sided",
            "Right two-sided",
        ]
        chi2_abacus = chi2_abacus.reindex(
            ["Left two-sided", "Left one-sided", "Right one-sided", "Right two-sided"],
            axis=1,
        ).reset_index()
        for key in chi2_abacus:
            chi2_abacus[key] = chi2_abacus[key].apply("{:.4g}".format)
        chi2_abacus = chi2_abacus.rename(columns={"alpha": r"$\alpha$"})
        chi2_abacus = cls.serialize(chi2_abacus, table_mode=table_mode)

        context = context | {
            "figure_mode": figure_mode,
            "table_mode": table_mode,
            "fit_payload": fit,
            "loss_payload": loss,
            "chi2_payload": chi2,
            "k2s_payload": k2s,
            "table_payload": table,
            "n": solver.k,
            "k": solver.k,
            "m": solver.m,
            "nu": solver.dof,
            "equation": solver._model_equation,
            "equation_array": solver._equation_array,
            "parameters": parameters,
            "solved": solver.solved(),
            "message": solver._solution["message"],
            "chi2_significant": solver._gof["pvalue"] >= 0.05,
            "chi2_statistic": "%.4g" % solver._gof["statistic"],
            "chi2_pvalue": "%.4f" % solver._gof["pvalue"],
            "chi2_abacus": chi2_abacus,
            "k2s_significant": solver._k2s["pvalue"] >= 0.05,
            "k2s_statistic": "%.4g" % solver._k2s["statistic"],
            "k2s_pvalue": "%.4f" % solver._k2s["pvalue"],
        }
        return context


class KineticSolverReportProcessor(ReportProcessor):
    template = "kinetic.md"

    @classmethod
    def context(
        cls,
        solver,
        context=None,
        table_mode="latex",
        figure_mode="svg",
        substance_indices=None,
    ):
        if context is None:
            context = {
                "title": "Kinetic Report",
                "author": "SciFit automatic report",
                "supervisor": "Jean Landercy",
            }

        if substance_indices is None:
            substance_indices = list(range(solver.k))

        context["equations"] = solver.model_equations()

        data = solver.coefficients()
        data.columns = data.columns.map(lambda x: "$%s$" % x)
        table = cls.serialize(data, table_mode=table_mode)
        context["coefficients"] = table

        data = solver.concentrations().set_index("").T
        data.index = data.index.map(lambda x: "${%s}$" % x)
        data["steady"] = data["steady"].apply(lambda x: "{%s}" % x)
        data = data.rename(columns={"x0": "$x_0$", "steady": "{Steady}"}).reset_index()
        table = cls.serialize(data, table_mode=table_mode)
        context["concentrations"] = table

        data = solver.constants()
        data = data.rename(
            columns={
                "k0": r"$k_{0}^{\rightarrow}$",
                "k0inv": r"$k_{0}^{\leftarrow}$",
            }
        )
        table = cls.serialize(data, table_mode=table_mode)
        context["constants"] = table

        axe = solver.plot_solve(substance_indices=substance_indices)
        figure = cls.serialize(axe)
        plt.close(axe.figure)
        context["solution"] = figure

        axe = solver.plot_quotients()
        figure = cls.serialize(axe)
        plt.close(axe.figure)
        context["quotients"] = figure

        axe = solver.plot_selectivities(substance_indices=substance_indices)
        figure = cls.serialize(axe)
        plt.close(axe.figure)
        context["selectivities"] = figure

        axe = solver.plot_global_selectivities(substance_indices=substance_indices)
        figure = cls.serialize(axe)
        plt.close(axe.figure)
        context["global_selectivities"] = figure

        context["mode"] = solver._mode
        context["tmin"] = solver._solution.t.min()
        context["tmax"] = solver._solution.t.max()
        context["dt"] = np.diff(solver._solution.t)[0]
        context["nt"] = solver._solution.t.size

        context["n"] = solver.n
        context["k"] = solver.k

        data = solver.dataset().fillna("").set_index("t")
        n = data.shape[0] // 50
        data = data.iloc[::n, :]
        # columns = data.filter(regex="d.+/d.")
        # data = data.drop(columns, axis=1)

        table = cls.serialize(
            data.filter(regex="^[A-Z]{1}$").reset_index(), table_mode=table_mode
        )
        context["data_concentrations"] = table

        table = cls.serialize(
            data.filter(regex="^d[A-Z]{1}/dt$").reset_index(), table_mode=table_mode
        )
        context["data_rates"] = table

        table = cls.serialize(
            data.filter(regex="^Q[0-9]+$").reset_index(), table_mode=table_mode
        )
        context["data_quotients"] = table

        return context


class ChromatogramSolverReportProcessor(ReportProcessor):
    template = "chromatography.md"

    @classmethod
    def context(
        cls,
        solver,
        context=None,
        table_mode="latex",
        figure_mode="svg",
        substance_indices=None,
    ):
        if context is None:
            context = {
                "title": "Chromatography Report",
                "author": "SciFit automatic report",
                "supervisor": "Jean Landercy",
            }

        axe = solver.plot_fit()
        figure = cls.serialize(axe)
        plt.close(axe.figure)
        context["chromatogram"] = figure

        data = solver.summary()
        # data = data.rename(
        #     columns={
        #         "k0": r"$k_{0}^{\rightarrow}$",
        #         "k0inv": r"$k_{0}^{\leftarrow}$",
        #     }
        # )
        table = cls.serialize(data, table_mode=table_mode)
        context["summary"] = table

        return context
