import base64
import io
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

import jinja2
import matplotlib.pyplot as plt
import pandas as pd
import pypandoc


class ReportProcessor:

    @staticmethod
    def render(context=None, template="base.html", directory="scifit/toolbox/resources/reports/"):
        loader = jinja2.FileSystemLoader(searchpath=directory)
        environment = jinja2.Environment(loader=loader)
        template = environment.get_template(template)
        payload = template.render(**(context or {}))
        return payload

    @staticmethod
    def convert(payload, file="report", path=".cache/media/reports", mode="pdf"):
        filename = pathlib.Path(path) / file
        pypandoc.convert_text(payload, mode, format="html", outputfile="%s.%s" % (filename, mode))

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
            item.to_html(stream, float_format="{:.3g}".format)
            return stream.getvalue()
