import os
import sys

# -- Project information -----------------------------------------------------
project = "GeoGenIE"  # Name of the project

copyright = "2024, Bradley T. Martin and Tyler K. Chafin"
author = "Drs. Bradley T. Martin and Tyler K. Chafin"
release = "1.0.4"  # Version of the project
version = release  # Version for the documentation

# -- Path setup --------------------------------------------------------------
# Add the project's root directory to sys.path
sys.path.insert(0, os.path.abspath("../../../"))

# -- Sphinx Extensions -------------------------------------------------------
# Add extensions for autodoc, type hints, and more
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Supports Google-style docstrings
    "sphinx_autodoc_typehints",  # Type hints in function signatures
    "sphinx.ext.todo",  # To-do directives in documentation
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinxcontrib.bibtex",  # For bibliography management
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
]

# Link to sklearn's documentation for intersphinx
# This allows cross-referencing between different documentation projects
intersphinx_mapping = {
    "sklearn": (
        "https://scikit-learn.org/stable",
        (None, "./_intersphinx/sklearn-objects.inv"),
    )
}

bibtex_bibfiles = ["./references.bib"]  # Path to the bibliography file

# Enable displaying todos
todo_include_todos = True

# -- HTML output theme and customization -------------------------------------
html_theme = "sphinx_rtd_theme"  # Read the Docs theme

html_context = {
    "display_github": True,  # Enable GitHub integration
    "github_user": "btmartin721",  # GitHub username
    "github_repo": "GeoGenIE",  # GitHub repo
    "github_version": "master",  # Branch to use
    "current_version": "v1.0.4",  # Project version
    "display_version": True,  # Display version number in the theme
    "latest_version": "master",  # Define the latest stable version
    "display_edit_on_github": True,  # Add 'Edit on GitHub' link
}

# Custom logo and favicon
html_logo = "../../../img/geogenie_logo.png"

# -- General configuration ---------------------------------------------------
# Files or directories to ignore during build
exclude_patterns = ["**/setup.rst", "**/tests.rst", "_build", "Thumbs.db", ".DS_Store"]
