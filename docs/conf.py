import pydiffsol

project = 'pydiffsol'
copyright = '2025, Alex Allmont'
author = 'Alex Allmont'
release = pydiffsol.version()
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    'sphinx_rtd_theme',
]

autosummary_generate = True
autodoc_member_order = 'groupwise'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_rtd_theme'
