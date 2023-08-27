import base64
import io
import pathlib
from collections.abc import Iterable

import pandas as pd
import matplotlib.pyplot as plt

import jinja2
import matplotlib.pyplot as plt
import pandas as pd
import pypandoc


class ReportProcessor:

    @staticmethod
    def render(context=None, template="base.md", directory="scifit/toolbox/resources/reports/"):
        loader = jinja2.FileSystemLoader(searchpath=directory)
        environment = jinja2.Environment(loader=loader)
        template = environment.get_template(template)
        payload = template.render(**(context or {}))
        return payload

    @staticmethod
    def convert(payload, file="report", path=".cache/media/reports", mode="pdf"):
        filename = pathlib.Path(path) / file
        pypandoc.convert_text(
            payload, mode, format="md", outputfile="%s.%s" % (filename, mode),
            #filters=["citeproc"],
            extra_args=["--pdf-engine=pdflatex", "--biblatex"],
        )

    @staticmethod
    def serialize(item):

        if isinstance(item, plt.Axes):
            stream = io.BytesIO()
            item.figure.savefig(stream, format="svg")
            payload = "data:image/svg+xml;base64,{}".format(
                base64.b64encode(stream.getvalue()).decode()
            )
            return payload

        elif isinstance(item, pd.DataFrame):
            stream = io.StringIO()
            item.to_markdown(stream, index=False) #, float_format="{:.3g}".format)
            return stream.getvalue()

    def report(self, solver, context=None, path=".cache/media/reports", file="report", mode="pdf"):

        if context is None:
            context = {
                "title": "Fit Report",
                "author": "SciFit automatic report",
                "supervisor": "Jean Landercy",
            }

        axe = solver.plot_fit()
        fit = self.serialize(axe)
        plt.close(axe.figure)

        axe = solver.plot_loss()
        if isinstance(axe, Iterable):
            axe = axe[0][0]
        loss = self.serialize(axe)
        plt.close(axe.figure)

        axe = solver.plot_chi_square()
        chi2 = self.serialize(axe)
        plt.close(axe.figure)

        axe = solver.plot_kolmogorov()
        k2s = self.serialize(axe)
        plt.close(axe.figure)

        data = solver.dataset().reset_index(drop=True).round(3)
        data = data.drop(["yerrrel", "yerrabs"], axis=1)
        data["pm"] = r"$\pm$"
        data = data.reindex(["x0", "y", "sy", "yhat", "yerr", "yerrsqr", "chi2"], axis=1)
        data = data.rename(columns={
            "x0": r"$x_0$",
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
        table = self.serialize(data)

        parameters = solver.parameters().reset_index()
        parameters["pm"] = r"$\pm$"
        parameters = parameters.reindex(["index", "b", "pm", "sb"], axis=1)
        parameters = parameters.rename(columns={
            "index": r"$i$",
            "b": r"$\beta_i$",
            "pm": r"$\pm$",
            "sb": r"$\sigma_{\beta_i}$"
        })
        parameters = self.serialize(parameters)

        context = context | {
            "fit_payload": fit,
            "loss_payload": loss,
            "chi2_payload": chi2,
            "k2s_payload": k2s,
            "table_payload": table,
            "n": solver.n,
            "k": solver.k,
            "m": solver.m,
            "equation": solver._model_equation,
            "parameters": parameters,
            "solved": solver.solved(),
            "message": solver._solution["message"],
            "chi2_significant": solver._gof["pvalue"] >= 0.05,
            "chi2_pvalue": "%.4f" % solver._gof["pvalue"],
            "k2s_significant": solver._k2s["pvalue"] >= 0.05,
            "k2s_pvalue": "%.4f" % solver._k2s["pvalue"],
        }

        payload = self.render(context)
        self.convert(payload, path=path, file=file, mode=mode)
