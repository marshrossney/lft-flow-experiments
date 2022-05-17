import nox

nox.options.session = []


@nox.session
def lint(session):
    session.install("flake8", "black")
    session.run("black", ".")
    session.run("flake8", ".")


@nox.session
def test(session):
    session.install("pytest")
    session.run("pytest", "--pyargs", "convnet_phi4")
