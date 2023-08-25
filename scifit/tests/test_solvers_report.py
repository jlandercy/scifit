import os
from unittest import TestCase

import numpy as np

import pypandoc

#
# class TestSimple(TestCase):
#
#     def test_export_changes_pdf(self):
#         pypandoc.convert_file('CHANGES.md', 'pdf', outputfile=".cache/changes.pdf")
#
#     def test_export_readme_pdf(self):
#         pypandoc.convert_file('scifit/tests/resources/reports/sample/report.md', 'pdf', outputfile=".cache/report.pdf")

#
# class ReportProcessor:
#
#     directory = None
#     report = "report"
#
#     def setUp(self):
#         self.current = os.getcwd()
#         os.chdir(self.directory)
#
#     def tearDown(self):
#         os.chdir(self.current)
#
#     def test_report_pdf(self):
#         pypandoc.convert_file(self.report + ".md", 'pdf', outputfile="%s.pdf" % self.report)
#
#
# class TestSimpleReport(ReportProcessor, TestCase):
#     directory = "scifit/tests/resources/reports/sample"

