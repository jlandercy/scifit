import base64
import io
import pathlib

import matplotlib.pyplot as plt

import jinja2
import matplotlib.pyplot as plt
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
        pypandoc.convert_text(payload, mode, format="md", outputfile="%s.pdf" % filename)

    @staticmethod
    def serialize(item):
        if isinstance(item, plt.Axes):
            stream = io.BytesIO()
            item.figure.savefig(stream, format="svg")
            stream.seek(0)
            payload = "data:image/svg+xml;base64,{}".format(
                base64.b64encode(stream.read()).decode()
            )
            return payload
