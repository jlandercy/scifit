import base64
import io
import pathlib
from collections.abc import Iterable

import pandas as pd
import matplotlib.pyplot as plt

import jinja2
import pypandoc

from scifit.errors.base import *


class ReportProcessor:

    @staticmethod
    def render(context=None, template="base.md", directory=None):
        if directory is None:
            directory = pathlib.Path(__file__).parent / "resources/reports/"
        loader = jinja2.FileSystemLoader(searchpath=str(directory))
        environment = jinja2.Environment(loader=loader)
        template = environment.get_template(template)
        payload = template.render(**(context or {}))
        return payload

    @staticmethod
    def convert(payload, file="report", path=".", mode="pdf"):
        filename = pathlib.Path(path) / file
        pypandoc.convert_text(
            payload, mode, format="md", outputfile="%s.%s" % (filename, mode),
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
            frame.to_latex(stream, index=index, header=True, longtable=True, column_format="S" * frame.shape[1])
        elif mode == "md":
            frame.to_markdown(stream, index=index)
        elif mode == "html":
            frame.to_html(stream, index=index)
        return stream.getvalue()

    @staticmethod
    def serialize(item, table_mode="latex"):

        if isinstance(item, plt.Axes):
            return ReportProcessor.serialize_figure(item)

        elif isinstance(item, pd.DataFrame):
            return ReportProcessor.serialize_table(item, mode=table_mode)

    @staticmethod
    def context(solver, context=None, table_mode="latex", figure_mode="svg"):

        if context is None:
            context = {
                "title": "Fit Report",
                "author": "SciFit automatic report",
                "supervisor": "Jean Landercy",
            }

        axe = solver.plot_fit()
        fit = ReportProcessor.serialize(axe)
        plt.close(axe.figure)

        axe = solver.plot_loss()
        if isinstance(axe, Iterable):
            axe = axe[0][0]
        loss = ReportProcessor.serialize(axe)
        plt.close(axe.figure)

        axe = solver.plot_chi_square()
        chi2 = ReportProcessor.serialize(axe)
        plt.close(axe.figure)

        axe = solver.plot_kolmogorov()
        k2s = ReportProcessor.serialize(axe)
        plt.close(axe.figure)

        # Dataset
        data = solver.dataset()
        data = data.drop(["yerrrel", "yerrabs"], axis=1)
        # Too bad pandas fails now for formatting
        for key in data:
            data[key] = data[key].apply("{:.4g}".format)
        data = data.reset_index()
        data = data.reindex(["id", "x0", "y", "sy", "yhat", "yerr", "chi2"], axis=1)
        data = data.rename(columns={
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
        })
        table = ReportProcessor.serialize(data, table_mode=table_mode)

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
        parameters = parameters.rename(columns={
            "index": r"$i$",
            "b": r"$\beta_i$",
            "pm": r"$\pm$",
            "sb": r"$\sigma_{\beta_i}$"
        })
        parameters = ReportProcessor.serialize(parameters, table_mode=table_mode)

        # Chi 2 Abacus
        chi2_abacus = solver.chi_square_table()
        chi2_abacus = chi2_abacus.groupby(["alpha", "key"])[["low-value", "high-value"]].first().unstack().dropna(how="all", axis=1)
        chi2_abacus = chi2_abacus.droplevel(0, axis=1)
        chi2_abacus.columns = ["Left-sided", "Left two-sided", "Right-sided", "Right two-sided"]
        chi2_abacus = chi2_abacus.reindex(["Left two-sided", "Left-sided", "Right-sided", "Right two-sided"], axis=1).reset_index()
        for key in chi2_abacus:
            chi2_abacus[key] = chi2_abacus[key].apply("{:.4g}".format)
        chi2_abacus = chi2_abacus.rename(columns={
            "alpha": r"$\alpha$"
        })
        chi2_abacus = ReportProcessor.serialize(chi2_abacus, table_mode=table_mode)

        context = context | {
            "figure_mode": figure_mode,
            "table_mode": table_mode,
            "fit_payload": fit,
            "loss_payload": loss,
            "chi2_payload": chi2,
            "k2s_payload": k2s,
            "table_payload": table,
            "n": solver.n,
            "k": solver.k,
            "m": solver.m,
            "nu": solver.dof,
            "equation": solver._model_equation,
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

    @staticmethod
    def report(solver, context=None, path=".", file="report", mode="pdf"):
        modes = {"pdf", "html", "docx"}
        if mode not in modes:
            raise ConfigurationError("Mode must be in %s, got '%s' instead" % (modes, mode))
        if mode == "pdf":
            table_mode = "latex"
        elif mode == "html":
            table_mode = "html"
        else:
            table_mode = "md"
        context = ReportProcessor.context(solver, context=context, table_mode=table_mode)
        payload = ReportProcessor.render(context)
        ReportProcessor.convert(payload, path=path, file=file, mode=mode)
