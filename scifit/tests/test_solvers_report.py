from unittest import TestCase

from scifit.solvers import linear
from scifit.toolbox import report


class TestReportProcessor(TestCase):

    context = {
        "title": "Fit Report",
        "author": "SciFit automatic report",
        "supervisor": "Jean Landercy",
    }

    def setUp(self):
        self.processor = report.ReportProcessor()
        self.solver = linear.LinearFitSolver()
        self.data = self.solver.synthetic_dataset(sigma=0.15)
        self.solver.fit(self.data)

    def test_jinja_processor(self):
        payload = self.processor.render(self.context)

    def test_pandoc_converter(self):
        self.processor.convert("# Hello world\n##Foo\n###Bar", file="dummy")

    def test_serialize_figure(self):
        axe = self.solver.plot_fit()
        payload = self.processor.serialize(axe)

    def test_serialize_table(self):
        data = self.solver.dataset()
        payload = self.processor.serialize(data)

    def test_full_chain(self):

        axe = self.solver.plot_fit()
        fit = self.processor.serialize(axe)

        axe = self.solver.plot_loss()
        loss = self.processor.serialize(axe)

        axe = self.solver.plot_chi_square()
        chi2 = self.processor.serialize(axe)

        axe = self.solver.plot_kolmogorov()
        k2s = self.processor.serialize(axe)

        data = self.solver.dataset().reset_index(drop=True).round(3)
        table = self.processor.serialize(data)

        context = self.context | {
            "fit_payload": fit,
            "loss_payload": loss,
            "chi2_payload": chi2,
            "k2s_payload": k2s,
            "table_payload": table,
            "n": self.solver.n,
            "k": self.solver.k,
            "m": self.solver.m,
            "equation": self.solver._model_equation,
            "solved": self.solver.solved(),
            "message": self.solver._solution["message"]
        }

        payload = self.processor.render(context)
        self.processor.convert(payload, file="report", mode="pdf")
