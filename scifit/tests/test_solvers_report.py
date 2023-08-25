from unittest import TestCase

import numpy as np

import pypandoc


class TestReadme(TestCase):

    def test_export_changes_pdf(self):
        pypandoc.convert_file('CHANGES.md', 'pdf', outputfile=".cache/changes.pdf")

    def test_export_readme_pdf(self):
        pypandoc.convert_file('README.md', 'pdf', outputfile=".cache/readme.pdf")
