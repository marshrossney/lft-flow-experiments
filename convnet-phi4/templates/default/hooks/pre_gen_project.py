"""A hook script that runs before experiment generation.

This script will be executed in the generated experiment directory BEFORE
it is populated with files.

Templated variables, e.g. {{ cookiecutter._timestamp }} are rendered.

If it fails, the generation process will be aborted and the directory deleted.

Use it to validate the provided configuration, run tests etc.
"""
pass
