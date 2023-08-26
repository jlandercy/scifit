from unittest import TestCase

from scifit.solvers import linear
from scifit.toolbox import report


class TestBasicReportProcessor(TestCase):

    file = "report_basic"
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
        self.processor.convert("# Hello world\n\n##Foo\n\n###Bar", file="dummy")

    def test_serialize_figure(self):
        axe = self.solver.plot_fit()
        payload = self.processor.serialize(axe)

    def test_serialize_table(self):
        data = self.solver.dataset()
        payload = self.processor.serialize(data)

    def test_report_pdf(self):
        self.processor.report(self.solver, context=self.context, file=self.file, mode="pdf")

    def test_report_docx(self):
        self.processor.report(self.solver, context=self.context, file=self.file, mode="docx")

    def test_report_html(self):
        self.processor.report(self.solver, context=self.context, file=self.file, mode="html")


class TestBadReport(TestBasicReportProcessor):

    file = "report_bad"

    def setUp(self):
        super().setUp()
        self.solver = linear.ConstantFitSolver()
        self.solver.fit(self.data)
