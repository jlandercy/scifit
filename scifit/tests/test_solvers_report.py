from unittest import TestCase

from scifit.solvers import linear
from scifit.toolbox import report


class TestReportProcessor(TestCase):

    context = {
        "name": "Jean Landercy"
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
        figure = self.processor.serialize(axe)

        data = self.solver.dataset()#[["x0", "y", "sy"]].reset_index()
        table = self.processor.serialize(data)

        context = self.context | {
            "fit_payload": figure,
            "table_payload": table,
        }

        payload = self.processor.render(context)
        self.processor.convert(payload, file="report", mode="pdf")
