import json
import pathlib
import re

import nox
from lxml import etree

# Settings:

nox.options.envdir = ".cache"
nox.options.default_venv_backend = "none"
nox.options.sessions = ["tests", "coverage", "security", "safety", "linter", "syntax", "types", "styles"]

cache_path = pathlib.Path(nox.options.envdir)
cache_path.mkdir(exist_ok=True)

package_path = pathlib.Path(__file__).parent
package_name = package_path.parts[-1]
reports_path = pathlib.Path(nox.options.envdir) / 'reports'
reports_path.mkdir(exist_ok=True)


# Sessions:

@nox.session
def clean(session):
    """Package Code Cleaner"""
    report = reports_path / "clean.log"
    with report.open("w") as handler:
        session.run("python", "-m", "isort", ".", stdout=handler)
        session.run("python", "-m", "black", package_name, stdout=handler)


@nox.session
def package(session):
    """Package Builds (badge)"""
    report = reports_path / "package.log"
    with report.open("w") as handler:
        session.run("python", "setup.py", "sdist", "bdist_wheel", stdout=handler)
    badge = reports_path / 'package.svg'
    badge.unlink(missing_ok=True)
    session.run("anybadge", "--value=build", f"--file={badge:}", "--label=package")


@nox.session
def install(session):
    """Package Installer"""
    wheels = [str(file) for file in pathlib.Path("./dist").glob("*.whl")]
    if not wheels:
        session.error("No wheel found, first package then install")
    report = reports_path / "install.log"
    with report.open("w") as handler:
        session.run("python", "-m", "pip", "install", *wheels, stdout=handler)


@nox.session
def uninstall(session):
    """Package Uninstaller"""
    report = reports_path / "uninstall.log"
    with report.open("w") as handler:
        session.run("python", "-m", "pip", "uninstall", "-y", package_name, stdout=handler)


@nox.session
def tests(session):
    """Package Test Suite Report (badge)"""
    report = reports_path / "tests.log"
    with report.open("w") as handler:
        session.run("python", "-m", "xmlrunner", "--output-file", str(reports_path / "tests.xml"),
                    "discover", "-v", "tests", stdout=handler)
    pattern = re.compile(r"Ran (?P<count>[\d]+) tests in (?P<elapsed>[.\d]+)s")
    count, elapsed = pattern.findall(report.read_text())[0]
    badge = reports_path / 'tests.svg'
    badge.unlink(missing_ok=True)
    session.run("anybadge", f"--value={count:}/{elapsed:}s", f"--file={badge:}", "--label=tests")


@nox.session
def coverage(session):
    """Package Test Suite Coverage Report (badge)"""
    env = {"COVERAGE_FILE": str(reports_path / "coverage.dat")}
    report = reports_path / "coverage.xml"
    session.run("python", "-m", "coverage", "run", "-m", "unittest",
                "discover", "-v", "tests", env=env)
    session.run("python", "-m", "coverage", "report", "--omit=venv/**/*", env=env)
    session.run("python", "-m", "coverage", "xml", "-o", f"{report:}", env=env)
    with report.open() as handler:
        root = etree.XML(handler.read())
    score = float(root.get("line-rate"))*100.
    badge = reports_path / 'coverage.svg'
    badge.unlink(missing_ok=True)
    session.run("anybadge", f"--value={score:}", f"--file={badge:}", "coverage")


@nox.session
def security(session):
    """Package Security Report (badge)"""
    logs = reports_path / "security.log"
    report = reports_path / "security.json"
    with logs.open("w") as handler:
        session.run("bandit", "-v", "-f", "json", "-o", str(report),
                    "-r", package_name, stdout=handler)
    with report.open() as handler:
        results = json.load(handler)["metrics"]["_totals"]
    for key in results.keys():
        if key.startswith('SEVERITY'):
            value = int(results[key])
            level = key.split(".")[-1].lower()
            label = f"security-{level:}"
            badge = reports_path / f"{label:}.svg"
            badge.unlink(missing_ok=True)
            session.run("anybadge", f"--value={value:}", f"--file={badge:}", f"--label={label:}",
                        "1=green", "2=red")


@nox.session
def safety(session):
    """Package Safety Report (badge)"""
    logs = reports_path / "security.log"
    report = reports_path / "safety.json"
    with logs.open("w") as handler:
        session.run("safety", "check", "--full-report", "--json",
                    "--output", str(report), stdout=handler)
    with report.open() as handler:
        results = json.load(handler)
    count = len(results)
    badge = reports_path / "safety.svg"
    badge.unlink(missing_ok=True)
    session.run("anybadge", f"--value={count:}", f"--file={badge:}", f"--label=safety",
                "1=green", "2=red")


@nox.session
def linter(session):
    """Package Linter Report (badge)"""
    report = reports_path / "linter.log"
    with report.open("w") as handler:
        session.run(
            "pylint", package_name,
            "--output-format=parseable", "--rcfile=.pylintrc",
            "--fail-under=6", stdout=handler
        )
    pattern = re.compile(r"Your code has been rated at (?P<score>[-.\d]*)/10")
    score = float(pattern.findall(report.read_text())[0])
    badge = reports_path / 'linter.svg'
    badge.unlink(missing_ok=True)
    session.run("anybadge", f"--value={score:}/10", f"--file={badge:}", "--label=linter", "pylint")


@nox.session
def syntax(session):
    """Package Syntax Report (badge)"""
    logs = reports_path / "syntax.log"
    report = logs.with_suffix('.xml')
    logs.unlink(missing_ok=True)
    session.run("flake8", "-v", "--tee", "--exit-zero", "--output-file", str(logs), package_name)
    session.run("flake8_junit", str(logs), str(report))
    with report.open("rb") as handler:
        root = etree.XML(handler.read())
    for key in ["errors", "failures"]:
        value = root.get(key)
        label = f"syntax-{key:}"
        badge = reports_path / f"{label:}.svg"
        badge.unlink(missing_ok=True)
        session.run("anybadge", f"--value={value:}", f"--file={badge:}", f"--label={label:}",
                    "1=green", "2=red")


@nox.session
def types(session):
    """Package Type Hints Report (badge)"""
    report = reports_path / "types.log"
    # Result of mypy is cached and differential:
    with report.open("w") as handler:
        session.run("python", "-m", "mypy",
                    "--cache-dir", str(reports_path.parent / ".mypy"),
                    "-v", package_name, stdout=handler)
    pattern = re.compile(r"Build finished in (?P<elapsed>[.\d]+) seconds "
                          "with (?P<modules>[\d]+) modules, "
                          "and (?P<errors>[\d]+) errors\n(?P<status>[\w]+): "
                          "no issues found in (?P<sources>[\d]+) source files")
    *_, status, files = pattern.findall(report.read_text())[0]
    badge = reports_path / 'types.svg'
    badge.unlink(missing_ok=True)
    session.run("anybadge", f"--value={status:}", f"--file={badge:}", "--label=type-hints")


@nox.session
def styles(session):
    """Package Code Styles Report (badge)"""
    report = reports_path / "styles.log"
    with report.open("w") as handler:
        session.run("python", "-m", "isort", "--diff", ".", stdout=handler)
        session.run("python", "-m", "black", "--check", "--diff", package_name,
                    stdout=handler, success_codes=[0, 1])
    pattern = re.compile(r"(?P<count>[\d]+) files would be reformatted")
    result = pattern.findall(report.read_text())
    badge = reports_path / 'styles.svg'
    badge.unlink(missing_ok=True)
    if result:
        count = result[0]
        session.run("anybadge", f"--value={count:}", f"--file={badge:}", "--color=red", "--label=code-style")
    else:
        session.run("anybadge", f"--value=black", f"--file={badge:}", "--color=black", "--label=code-style")


@nox.session
def notebooks(session):
    """Package Notebooks (badge)"""
    report = reports_path / "notebooks.log"
    with report.open("w") as handler:
        session.run("python", "-m", "ipykernel", "install", "--name=venv", stderr=handler)
        session.run(
            "python", "-m",
            "jupyter", "nbconvert", "--debug",
            "--ExecutePreprocessor.timeout=600",
            "--ExecutePreprocessor.kernel_name=venv",
            "--inplace", "--clear-output", "--to", "notebook",
            "--execute", "./docs/source/notebooks/*.ipynb",
            stderr=handler,
            success_codes=[0, 1]
        )
    with report.open() as handler:
        session.log(handler.read())
    pattern = re.compile(r"Writing (?P<bytes>[\d]+) bytes to")
    count = len(pattern.findall(report.read_text()))
    badge = reports_path / 'notebooks.svg'
    badge.unlink(missing_ok=True)
    session.run("anybadge", f"--value={count:}", f"--file={badge:}", "--label=notebooks",
                "1=red", "2=orange", "3=green")


@nox.session
def docs(session):
    """Package Documentation (badge)"""
    report = reports_path / "docs.log"
    with report.open("w") as handler:
        session.run("sphinx-build", "-b", "html", f"docs/source", str(cache_path / "docs"),
                    stdout=handler)
    pattern = re.compile(r"build (?P<status>[\w]+).")
    status = pattern.findall(report.read_text())[0]
    badge = reports_path / 'docs.svg'
    if badge.exists():
        badge.unlink()
    session.run("anybadge", f"--value={status:}", f"--file={badge:}", "--label=docs")
