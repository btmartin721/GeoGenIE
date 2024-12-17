# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
            
sys.path.insert(0, os.path.abspath('../'))

project = 'GeoGenIE'
copyright = '2024, Bradley T. Martin and Tyler K. Chafin'
author = 'Bradley T. Martin and Tyler K. Chafin'
release = '1.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.githubpages',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.bibtex',
]

# -- Link intersphinx to the scikit-learn documentation ------------------------
# -- Avoids a the metadata_routing error --------------------------------------
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Path to the BibTeX file
bibtex_bibfiles = ["references.bib"]

# Use "author_year" citation style for parentheses (e.g., (Akiba et al., 2020))
bibtex_reference_style = "super"

# Set bibliography output style to plain
bibtex_default_style = "plain"

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True
