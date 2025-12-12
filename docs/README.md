# pydiffsol docs

These docs use the Sphinx Napoleon extension to build Python API readthedocs directly from Rust code.

To test locally, build from the docs folder:

```shell
pip install sphinx
pip install -r docs/requirements.txt
cd docs
sphinx-build -b html . _build/html
```