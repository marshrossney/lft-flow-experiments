import nox

nox.options.session = []


@nox.session
def lint(session):
    session.install("flake8", "black")
    session.run("black", ".")
    session.run("flake8", ".")
