
# Note on CLI Availability

The `[tool.poetry.scripts]` entry point in `pyproject.toml` only becomes available after the package is installed. It's a Python packaging mechanism — Poetry/pip creates the `colony-env` wrapper script in the virtualenv's `bin/` directory during `poetry install` or `pip install`.

To make it available:

```shell
cd colony && poetry install
```

After that, `colony-env` will be on `PATH` within the Poetry virtualenv (or `poetry run colony-env` if you're not in the shell).

Alternatively, without installing, you can run it directly:

```shell
python -m colony.cli.deploy.cli --help
```

That said — this is the standard approach for Python CLI tools distributed as packages (same pattern as `pytest`, `black`, `ruff`, etc.). When users `pip install colony`, the `colony-env` command gets created automatically.

