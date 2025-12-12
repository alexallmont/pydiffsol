# pydiffsol docs

These docs use the Sphinx Napoleon extension to build Python API readthedocs directly from Rust code.

To test locally, build from the docs folder:

```shell
pip install -r docs/requirements.txt
cd docs
sphinx-build -b html . _build/html
```

Docs build from the latest published pydiffsol by default. If you have code changes you want to see in docs locally,
re-run `maturin develop`. This will re-install your local pydiffsol in your venv, then rebuild the docs.
